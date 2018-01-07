""" Defines the Basis object and supporting functions """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

from functools    import partial
from functools   import wraps
from itertools    import product

import copy as _copy
import numbers as _numbers
import collections as _collections

from numpy.linalg import inv as _inv
import numpy as _np

import math

from .basisconstructors import _basisConstructorDict
from .basisconstructors import cache_by_hashed_args
from .dim import Dim


class Basis(object):
    '''
    Encapsulates a basis

    A Basis stores groups of matrices, which are used to create/recast matrices and vectors used in pyGSTi.

    There are three different bases that GST can use and convert between (as well as the qutrit basis, not mentioned): 
      - The Standard ("std") basis:
         State space is the tensor product of [0,1] for each qubit, e.g. for two qubits: ``[00,01,10,11] = [ |0>|0>, |0>|1>, ... ]``
         the gate space is thus the tensor product of two qubit spaces, so identical in form to state space
         for twice qubits, but interpret as ket/bra states.  E.g. for a *one* qubit gate, std basis is: = ``[ |0><0|, |0><1|, ... ]``

      - The Pauli-product ("pp") basis:
         Not used for state space - just for gates.  Basis consists of tensor products of the 4 pauli matrices (normalized by sqrt(2)).
         Examples:

         - 1-qubit gate basis is [ I, X, Y, Z ]  (in std basis, each is a pauli mx / sqrt(2))
         - 2-qubit gate basis is [ IxI, IxX, IxY, IxZ, XxI, ... ] (16 of them. In std basis, each is the tensor product of two pauli/sqrt(2) mxs)

      - The Gell-Mann ("gm") basis:
         Not used for state space - just for gates.  Basis consists of the Gell-Mann matrices of the given dimension (useful for dimensions that are not a power of 2)
         Examples:

         - 1-qubit gate basis is [ I, X, Y, Z ]  (in std basis, each is a pauli mx / sqrt(2)) -- SAME as Pauli-product!
         - 2-qubit gate basis is the 16 Gell-Mann matrices of dimension 4. In std basis, each is as given by Wikipedia page up to normalization.

    Notes:
      - The elements of each basis are normalized so that Tr(Bi Bj) = delta_ij
      - since density matrices are Hermitian and all Gell-Mann and Pauli-product matrices are Hermitian too,
        gate parameterization by Gell-Mann or Pauli-product matrices have *real* coefficients, whereas
        in the standard basis gate matrices can have complex elements but these elements are additionally
        constrained.  This makes gate matrix parameterization and optimization much more convenient
        in the "gm" or "pp" bases.
    '''
    DefaultInfo = dict()
    CustomCount = 0 # The number of custom bases, used for serialized naming

    def __init__(self, name=None, dim=None, matrices=None, longname=None, real=None, labels=None):
        '''
        Initialize a basis object.

        Parameters
        ----------
        name : string or Basis
            Name of the basis to be created or a Basis to copy from
            if the name is 'pp', 'std', 'gm', or 'qt' and a dimension is provided, 
            then a default basis is created

        dim : int or list of ints,
            dimension/blockDimensions of the basis to be created.
            Only required when creating default bases

        matrices : list of numpy arrays, list of lists of numpy arrays, list of Basis objects/tuples
            Flexible argument that allows different types of basis creation
            When a list of numpy arrays, creates a non composite basis
            When a list of lists of numpy arrays or list of other bases, creates a composite bases with each outer list element as a composite part.

        longname : str
            Printout name for the basis

        real : bool
            Determine whether the basis admits complex elements during basis change

        labels : list of strings
            Labels for the basis matrices (i.e. I, X, Y, Z for the Pauli 2x2 basis)
        '''
        self.name, self._blockMatrices = _build_block_matrices(name, dim, matrices)
        self._matrices = self.get_composite_matrices() # Equivalent to matrices for non-composite bases

        for block in self._blockMatrices:
            for mx in block:
                mx.flags.writeable = False
        for mx in self._matrices:
            mx.flags.writeable = False

        def get_info(attr, default):
            """ Shorthand for retrieving a default value from the
                _basisConstructorDict dict """
            try:
                return getattr(_basisConstructorDict[self.name], attr)
            except KeyError:
                return default

        if real is None:
            real     = get_info('real', default=True)
        if longname is None:
            longname = get_info('longname', default=self.name)

        blockDims = [int(math.sqrt(len(group))) for group in self._blockMatrices]


        if labels is None:
            try:
                labels = basis_element_labels(self.name, blockDims)
            except NotImplementedError:
                labels = []
                for i, block in enumerate(self._blockMatrices):
                    for j in range(len(block)):
                        labels.append('M({})[{}]'.format(
                            self.name,
                            '{},{}'.format(i, j)
                            ))

        self.dim      = Dim(blockDims)
        self.labels   = labels
        self.longname = longname
        self.real     = real

    def copy(self):
        """Make a copy of this Basis object."""
        return Basis(self, longname=self.longname, real=self.real, labels=self.labels)

    def __str__(self):
        return '{} Basis : {}'.format(self.longname, ', '.join(self.labels))

    def __getitem__(self, index):
        return self._matrices[index]

    def __len__(self):
        return len(self._matrices)

    def __eq__(self, other):
        if isinstance(other, Basis):
            return _np.array_equal(self._matrices, other._matrices)
        else:
            return _np.array_equal(self._matrices, other)

    def __hash__(self):
        return hash((self.name, self.dim))

    def transform_matrix(self, to_basis):
        '''
        Retrieve a list of matrices by index 

        Parameters
        ----------
        index : int
            the position of matrices to retrieve

        Returns
        -------
        matrix to transform from this basis to another
        '''
        to_basis = Basis(to_basis, self.dim.blockDims)
        return _np.dot(to_basis.get_from_std(), self.get_to_std())

    def get_sub_basis_matrices(self, index):
        '''
        Retrieve a list of matrices by index 

        Parameters
        ----------
        index : int
            the position of matrices to retrieve

        Returns
        -------
        list of matrices
        '''
        return self._blockMatrices[index]

    @cache_by_hashed_args
    def is_normalized(self):
        '''
        Check if a basis is normalized

        Returns
        -------
        bool
        '''
        for mx in self._matrices:
            t = _np.trace(_np.dot(mx, mx))
            t = _np.real(t)
            if t > 1e-6:
                return False
        return True

    def get_composite_matrices(self):
        '''
        Build the large composite matrices of a composite basis
        ie for std basis with dim [2, 1], build
        [[1 0 0]  [[0 1 0]  [[0 0 0]  [[0 0 0]  [[0 0 0]
         [0 0 0]   [0 0 0]   [1 0 0]   [0 1 0]   [0 0 0]
         [0 0 0]], [0 0 0]], [0 0 0]], [0 0 0]], [0 0 1]]
        For a non composite basis, this just returns the basis matrices

        Returns
        -------
        numpy array
            array of matrices, shape == (nMatrices, d, d) where
            d is the composite matrix dimension.
        '''
        nMxs = sum([len(mxs) for mxs in self._blockMatrices])
        length  = sum(mxs[0].shape[0] for mxs in self._blockMatrices)
        compMxs = _np.zeros( (nMxs, length, length), 'complex')
        i, start   = 0, 0

        for mxs in self._blockMatrices:
            d = mxs[0].shape[0]
            for mx in mxs:
                compMxs[i][start:start+d,start:start+d] = mx
                i += 1
            start += d 
        assert(start == length and i == nMxs)
        return compMxs

    @cache_by_hashed_args
    def get_expand_mx(self):
        '''
        Retrieve the matrix that will convert from the direct sum space to the embedding space

        Returns
        -------
        numpy array
        '''
        # Dim: dmDim 5 gateDim 5 blockDims [1, 1, 1, 1, 1] embedDim 25

        x = sum(len(mxs) for mxs in self._blockMatrices)
        y = sum(mxs[0].shape[0] for mxs in self._blockMatrices) ** 2
        expandMx = _np.zeros((x, y), 'complex')
        for i, compMx in enumerate(self.get_composite_matrices()):
            flattened = compMx.flatten()
            assert len(flattened) == y, '{} != {}'.format(len(flattened), y)
            expandMx[i,0:y] = flattened 
        return expandMx

    @cache_by_hashed_args
    def get_contract_mx(self):
        '''
        Retrieve the matrix that will convert from the embedding space to the direct sum space,
        truncating if necessary (Currently without warning)

        Returns
        -------
        numpy array
        '''
        return self.get_expand_mx().T

    @cache_by_hashed_args
    def get_to_std(self):
        '''
        Retrieve the matrix that will convert from the current basis to the standard basis

        Returns
        -------
        numpy array
        '''
        toStd = _np.zeros((self.dim.gateDim, self.dim.gateDim), 'complex' )
        #Since a multi-block basis is just the direct sum of the individual block bases,
        # transform mx is just the transfrom matrices of the individual blocks along the
        # diagonal of the total basis transform matrix

        start = 0
        for mxs in self._blockMatrices:
            l = len(mxs)
            for j, mx in enumerate(mxs):
                toStd[start:start+l,start+j] = mx.flatten()
            start += l 
        assert(start == self.dim.gateDim)
        return toStd

    @cache_by_hashed_args
    def get_from_std(self):
        '''
        Retrieve the matrix that will convert from the standard basis to the current basis

        Returns
        -------
        numpy array
        '''
        return _inv(self.get_to_std())

    def equivalent(self, otherName):
        """ 
        Return a `Basis` of the type given by `otherName` and the dimensions
        of this `Basis`.
        
        Parameters
        ----------
        otherName : {'std', 'gm', 'pp', 'qt'}
            A standard basis abbreviation.

        Returns
        -------
        Basis
        """
        return Basis(otherName, self.dim.blockDims)

    def expanded_equivalent(self, otherName=None):
        """ 
        Return a single-block `Basis` of the type given by `otherName` and
        dimension given by the sum of the block dimensions of this `Basis`.
        
        Parameters
        ----------
        otherName : {'std', 'gm', 'pp', 'qt', None}
            A standard basis abbreviation.  If None, then this
            `Basis`'s name is used.

        Returns
        -------
        Basis
        """
        if otherName is None:
            otherName = self.name
        return Basis(otherName, sum(self.dim.blockDims))

    def std_equivalent(self):
        """ Convenience method identical to `.equivalent('std')` """
        return self.equivalent('std')

    def expanded_std_equivalent(self):
        """ Convenience method identical to `.expanded_equivalent('std')` """
        return self.expanded_equivalent('std')

def _build_composite_basis(bases):
    '''
    Build a composite basis from a list of tuples or Basis objects 
      (or a list of mixed tuples and Basis objects)

    Parameters
    ----------
    bases : list of tuples/Basis objects

    Returns
    -------
    Basis
        the composite basis created
    '''
    assert len(bases) > 0, 'Need at least one basis-dim pair to compose'
    basisObjs = []
    for item in bases:
        if isinstance(item, tuple):
            basisObjs.append(Basis(name=item[0], dim=item[1]))
        else:
            basisObjs.append(item)

    blockMatrices = [basis._matrices        for basis in basisObjs]
    name          = ','.join(basis.name     for basis in basisObjs)
    longname      = ','.join(basis.longname for basis in basisObjs)
    real          = all(basis.real          for basis in basisObjs)

    composite = Basis(matrices=blockMatrices, name=name, longname=longname, real=real)
    return composite


# Allow flexible basis building without cluttering the basis __init__ method with instance checking
def _build_block_matrices(name=None, dim=None, matrices=None):
    '''
    Build the block matrices for a basis object by flexible arguments

    Parameters
    ----------
    name : string or Basis
        Name of the basis to be created or a Basis to copy from
        if the name is 'pp', 'std', 'gm', or 'qt' and a dimension is provided, 
        then a default basis is created

    dim : int or list of ints,
        dimension/blockDimensions of the basis to be created.
        Only required when creating default bases

    matrices : list of numpy arrays, list of lists of numpy arrays, list of Basis objects/tuples
        Flexible argument that allows different types of basis creation
        When a list of numpy arrays, creates a non composite basis
        When a list of lists of numpy arrays or list of other bases, creates a composite bases with each outer list element as a composite part.

    Returns
    -------
    name, list of lists of numpy arrays
    '''
    if isinstance(name, Basis):
        basis         = name
        blockMatrices = _copy.deepcopy(basis._blockMatrices)
        name          = basis.name
    elif isinstance(name, list):
        basis = _build_composite_basis(name)
        blockMatrices = basis._blockMatrices
        name          = basis.name
    else:
        if matrices is None: # built by name and dim, ie Basis('pp', 4)
            assert name is not None, \
                    'If matrices is none, name must be supplied to Basis.__init__'
            matrices = _build_default_block_matrices(name, dim)

        if len(matrices) > 0:
            first = matrices[0]
            if isinstance(first, tuple) or \
                    isinstance(first, Basis):
                basis = _build_composite_basis(matrices) # really list of Bases or basis tuples
                blockMatrices = basis._blockMatrices
                name          = basis.name
            elif isinstance(first, list) or \
                 (isinstance(first, _np.ndarray) and first.ndim == 3): # els of matrices are sub-bases, so
                blockMatrices = matrices                              # set directly equal to blockMatrices
            elif isinstance(first, _np.ndarray) and first.ndim ==2:  # matrices is a list of matrices 
                blockMatrices = [matrices]                           # so set as the first (& only) sub-basis-block
        else:
            blockMatrices = []
        if name is None:
            name = 'CustomBasis_{}'.format(Basis.CustomCount)
            Basis.CustomCount += 1
    return name, blockMatrices

def _build_default_block_matrices(name, dim):
    '''
    Build the default block matrices for a basis object 
    (i.e. std, pp, gm, or qt basis matrices at time of writing)

    Parameters
    ----------
    name : string
        Name of the basis to be created

    dim : int
        dimension of the basis to be created.
    Returns
    -------
    list of lists of numpy arrays
    '''
    if name == 'unknown':
        return []
    if name not in _basisConstructorDict:
        raise NotImplementedError('No instructions to create supposed \'default\' basis:  {} of dim {}'.format(
            name, dim))
    f = _basisConstructorDict[name].constructor
    blockMatrices = []
    dim = Dim(dim)
    for blockDim in dim.blockDims:
        blockMatrices.append(f(blockDim))
    return blockMatrices

def basis_matrices(nameOrBasis, dim):
    '''
    Get the elements of the specifed basis-type which
    spans the density-matrix space given by dim.

    Parameters
    ----------
    name : {'std', 'gm', 'pp', 'qt'} or Basis
        The basis type.  Allowed values are Matrix-unit (std), Gell-Mann (gm),
        Pauli-product (pp), and Qutrit (qt).  If a Basis object, then 
        the basis matrices are contained therein, and its dimension is checked to
        match dim.

    dim : int 
        The dimension of the density-matrix space.

    Returns
    -------
    list
        A list of N numpy arrays each of shape (dmDim, dmDim),
        where dmDim is the matrix-dimension of the overall
        "embedding" density matrix (the sum of dimOrBlockDims)
        and N is the dimension of the density-matrix space,
        equal to sum( block_dim_i^2 ).
    '''
    if isinstance(nameOrBasis, Basis):
        basis = nameOrBasis
        assert(basis.dim.dmDim == dim), "Basis object has wrong dimension ({}) for requested basis matrices ({})".format(
            basis.dim.dmDim, dim)
        return basis.get_composite_matrices()

    name = nameOrBasis
    if name not in _basisConstructorDict:
        raise NotImplementedError('No instructions to create supposed \'default\' basis:  {} of dim {}'.format(
            name, dim))
    f = _basisConstructorDict[name].constructor

    return f(dim)

def basis_longname(basis):
    """
    Get the "long name" for a particular basis,
    which is typically used in reports, etc.

    Parameters
    ----------
    basis : string or Basis object

    Returns
    -------
    string
    """
    if isinstance(basis, Basis):
        return basis.longname
    return _basisConstructorDict[basis].longname


def basis_element_labels(basis, dimOrBlockDims):
    """
    Returns a list of short labels corresponding to to the
    elements of the described basis.  These labels are
    typically used to label the rows/columns of a box-plot
    of a matrix in the basis.

    Parameters
    ----------
    basis : {'std', 'gm', 'pp', 'qt'}
        Which basis the gateset is represented in.  Allowed
        options are Matrix-unit (std), Gell-Mann (gm),
        Pauli-product (pp) and Qutrit (qt).  If the basis is
        not known, then an empty list is returned.

    dimOrBlockDims : int or list
        Dimension of basis matrices.  If a list of integers,
        then gives the dimensions of the terms in a
        direct-sum decomposition of the density
        matrix space acted on by the basis.

    Returns
    -------
    list of strings
        A list of length dim, whose elements label the basis
        elements.
    """
    if isinstance(basis, Basis):
        return basis.labels

    if dimOrBlockDims == 1 or (hasattr(dimOrBlockDims,'__len__')
        and len(dimOrBlockDims) == 1 and dimOrBlockDims[0] == 1):
        return [ "" ]       # Special case of single element basis, in which
                            # case we return a single label.

    #Note: the loops constructing the labels in this function
    # must be in-sync with those for constructing the matrices
    # in std_matrices, gm_matrices, and pp_matrices.
    _, _, blockDims = Dim(dimOrBlockDims)

    lblList = []
    start = 0
    if basis == "std":
        for blockDim in blockDims:
            for i in range(start,start+blockDim):
                for j in range(start,start+blockDim):
                    lblList.append( "(%d,%d)" % (i,j) )
            start += blockDim

    elif basis == "gm":
        if dimOrBlockDims == 2: #Special case of Pauli's
            lblList = ["I","X","Y","Z"]

        else:
            for i,blockDim in enumerate(blockDims):
                d = blockDim

                #labels for gm_matrices of dim "blockDim":
                lblList.append("I^{(%d)}" % i) #identity on i-th block

                #X-like matrices, containing 1's on two off-diagonal elements (k,j) & (j,k)
                lblList.extend( [ "X^{(%d)}_{%d,%d}" % (i,k,j)
                                  for k in range(d) for j in range(k+1,d) ] )

                #Y-like matrices, containing -1j & 1j on two off-diagonal elements (k,j) & (j,k)
                lblList.extend( [ "Y^{(%d)}_{%d,%d}" % (i,k,j)
                                  for k in range(d) for j in range(k+1,d) ] )

                #Z-like matrices, diagonal mxs with 1's on diagonal until (k,k) element == 1-d,
                # then diagonal elements beyond (k,k) are zero.  This matrix is then scaled
                # by sqrt( 2.0 / (d*(d-1)) ) to ensure proper normalization.
                lblList.extend( [ "Z^{(%d)}_{%d}" % (i,k) for k in range(1,d) ] )


    elif basis == "pp":
        if dimOrBlockDims == 2: #Special case of Pauli's
            lblList = ["I","X","Y","Z"]

        else:
            #Some extra checking, since list-of-dims not supported for pp matrices yet.
            def _is_integer(x):
                return bool( abs(x - round(x)) < 1e-6 )
            if isinstance(dimOrBlockDims, _numbers.Integral):
                dimOrBlockDims = [dimOrBlockDims]
            assert isinstance(dimOrBlockDims, _collections.Container)
            for i, dim in enumerate(dimOrBlockDims):
                nQubits = _np.log2(dim)
                if not _is_integer(nQubits):
                    raise ValueError("Dimension for Pauli tensor product matrices must be an integer *power of 2*")
                nQubits = int(round(nQubits))

                 

                basisLblList = [ ['I','X','Y','Z'] ]*nQubits
                if i == 0 and len(dimOrBlockDims) == 1:
                    for sigmaLbls in product(*basisLblList):
                        lblList.append(''.join(sigmaLbls))
                else:
                    for sigmaLbls in product(*basisLblList):
                        lblList.append('{}{}'.format(''.join(sigmaLbls), i))


    elif basis == "qt":
        assert dimOrBlockDims == 3 or (hasattr(dimOrBlockDims,'__len__')
                and len(dimOrBlockDims) == 1 and dimOrBlockDims[0] == 3)
        lblList = ['II', 'X+Y', 'X-Y', 'YZ', 'IX', 'IY', 'IZ', 'XY', 'XZ']

    else:
        raise NotImplementedError('Unknown basis {}'.format(basis))
    return lblList
