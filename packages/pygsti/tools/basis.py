from functools    import partial
from itertools    import product

import numbers as _numbers
import collections as _collections

from numpy.linalg import inv as _inv
import numpy as _np

import math

from .opttools import cache_by_hashed_args 
from .basisconstructors import _basisConstructorDict
from . import matrixtools as _mt

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

        # Shorthand for retrieving a default value from the _basisConstructorDict dict
        def get_info(attr, default):
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
        start = 0
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
        return Basis(otherName, self.dim.blockDims)

    def expanded_equivalent(self, otherName=None):
        if otherName is None:
            otherName = self.name
        return Basis(otherName, sum(self.dim.blockDims))

    def std_equivalent(self):
        return self.equivalent('std')

    def expanded_std_equivalent(self):
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

def transform_matrix(from_basis, to_basis, dimOrBlockDims=None):
    '''
    Compute the transformation matrix between two bases

    Parameters
    ----------
    from_basis : Basis or str
        Basis being converted from

    to_basis : Basis or str
        Basis being converted to

    dimOrBlockDims : int or list of ints
        if strings provided as bases, the dimension of basis to use.

    Returns
    -------
    Basis
        the composite basis created
    '''
    if dimOrBlockDims is None:
        assert isinstance(from_basis, Basis)
    else:
        from_basis = Basis(from_basis, dimOrBlockDims)
    return from_basis.transform_matrix(to_basis)

def build_basis_pair(mx, from_basis, to_basis):
    if isinstance(from_basis, Basis) and not isinstance(to_basis, Basis):
        to_basis = from_basis.equivalent(to_basis)
    elif isinstance(to_basis, Basis) and not isinstance(from_basis, Basis):
        from_basis = to_basis.equivalent(from_basis)
    else:
        dimOrBlockDims = int(round(_np.sqrt(mx.shape[0])))
        to_basis = Basis(to_basis, dimOrBlockDims)
        from_basis = Basis(from_basis, dimOrBlockDims)
    return from_basis, to_basis

def build_basis_for_matrix(mx, basis):
    dimOrBlockDims = int(round(_np.sqrt(mx.shape[0])))
    return Basis(basis, dimOrBlockDims)

def change_basis(mx, from_basis, to_basis, dimOrBlockDims=None, resize=None):
    """
    Convert a gate matrix from one basis of a density matrix space
    to another.

    Parameters
    ----------
    mx : numpy array
        The gate matrix (a 2D square array) in the `from_basis` basis.

    from_basis, to_basis: {'std', 'gm', 'pp', 'qt'} or Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    dimOrBlockDims : int or list of ints, optional
        Structure of the density-matrix space. If None, then assume
        mx operates on a single-block density matrix space,
        i.e. on K x K density matrices with K == sqrt( mx.shape[0] ).

    Returns
    -------
    numpy array
        The given gate matrix converted to the `to_basis` basis.
        Array size is the same as `mx`.
    """
    if len(mx.shape) not in [1, 2]:
        raise ValueError("Invalid dimension of object - must be 1 or 2, i.e. a vector or matrix")

    if dimOrBlockDims is None:
        dimOrBlockDims = int(round(_np.sqrt(mx.shape[0])))
        #assert( dimOrBlockDims**2 == mx.shape[0] )

    dim        = Dim(dimOrBlockDims)
    from_basis = Basis(from_basis, dim)
    to_basis   = Basis(to_basis, dim)
    #TODO: check for 'unknown' basis here and display meaningful warning - otherwise just get 0-dimensional basis...

    if from_basis.dim.gateDim != to_basis.dim.gateDim: 
        raise ValueError('Automatic basis expanding/contracting temporarily disabled')
        #or \
        #    resize is not None:
        if resize is None:
            assert len(to_basis.dim.blockDims) == 1 or len(from_basis.dim.blockDims) == 1, \
                    'Cannot convert from composite basis {} to another composite basis {}'.format(from_basis, to_basis)
            '''
            WRONG
            if from_basis.dim.gateDim < to_basis.dim.gateDim:
                resize = 'contract'
            else:
                resize = 'expand'
            '''
        return resize_mx(mx, resize=resize, startBasis=from_basis, endBasis=to_basis)

    if from_basis.name == to_basis.name:
        return mx.copy()

    toMx   = from_basis.transform_matrix(to_basis)
    fromMx = to_basis.transform_matrix(from_basis)

    isMx = len(mx.shape) == 2 and mx.shape[0] == mx.shape[1]
    if isMx:
        ret = _np.dot(toMx, _np.dot(mx, fromMx))
    else: # isVec
        ret = _np.dot(toMx, mx)

    if not to_basis.real:
        return ret

    if _np.linalg.norm(_np.imag(ret)) > 1e-8:
        raise ValueError("Array has non-zero imaginary part (%g) after basis change (%s to %s)!\n%s" %
                         (_np.linalg.norm(_np.imag(ret)), from_basis, to_basis, ret))
    return _np.real(ret)

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
        blockMatrices = basis._blockMatrices
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

def resize_std_mx(mx, resize, stdBasis1, stdBasis2):
    assert stdBasis1.dim.embedDim == stdBasis2.dim.embedDim
    if stdBasis1.dim.gateDim == stdBasis2.dim.gateDim:
        return mx
    #print('{}ing {} to {}'.format(resize, stdBasis1, stdBasis2))
    #print('Dims: ({} to {})'.format(stdBasis1.dim, stdBasis2.dim))
    if resize == 'expand':
        assert stdBasis1.dim.gateDim < stdBasis2.dim.gateDim
        right = _np.dot(mx, stdBasis1.get_expand_mx())
        mid   = _np.dot(stdBasis1.get_contract_mx(), right)
    elif resize == 'contract':
        assert stdBasis1.dim.gateDim > stdBasis2.dim.gateDim
        right = _np.dot(mx, stdBasis2.get_contract_mx())
        mid = _np.dot(stdBasis2.get_expand_mx(), right)
    return mid

def flexible_change_basis(mx, startBasis, endBasis):
    if startBasis.dim.gateDim == endBasis.dim.gateDim:
        return change_basis(mx, startBasis, endBasis)
    if startBasis.dim.gateDim < endBasis.dim.gateDim:
        resize = 'expand'
    else:
        resize = 'contract'
    stdBasis1 = startBasis.std_equivalent()
    stdBasis2 = endBasis.std_equivalent()
    start = change_basis(mx, startBasis, stdBasis1)
    mid   = resize_std_mx(mx, resize, stdBasis1, stdBasis2)
    end   = change_basis(mid, stdBasis2, endBasis)
    return end

def resize_mx(mx, dimOrBlockDims=None, resize=None, startBasis='std', endBasis='std'):
    if dimOrBlockDims is None:
        return mx
    if resize == 'expand':
        a = Basis('std', dimOrBlockDims)
        b = Basis('std', sum(dimOrBlockDims))
    else:
        a = Basis('std', sum(dimOrBlockDims))
        b = Basis('std', dimOrBlockDims)
    return resize_std_mx(mx, resize, a, b)


    '''
    Convert a gate matrix in an arbitrary basis of either a "direct-sum" or the embedding
    space to a matrix in the same basis in the opposite space.

    Parameters
    ----------
    mx: numpy array
        Matrix of size N x N, where N is the dimension
        of the density matrix space, i.e. sum( dimOrBlockDims_i^2 )

    dimOrBlockDims : int or list of ints
        Structure of the density-matrix space.

    Returns
    -------
    numpy array
        A M x M matrix, where M is the dimension of the
        embedding density matrix space, i.e.
        sum( dimOrBlockDims_i )^2
    '''
    bothBases = isinstance(startBasis, Basis) and isinstance(endBasis, Basis)
    if resize is not None and \
            bothBases or \
            (dimOrBlockDims is not None):
        if dimOrBlockDims is not None:# not bothBases:
            dim = Dim(dimOrBlockDims)
            startBasis = Basis(startBasis, dim)
            endBasis   = Basis(endBasis, dim)
        else:
            assert bothBases
        stdBasis1  = Basis('std', startBasis.dim.blockDims)
        stdBasis2  = Basis('std', endBasis.dim.blockDims)
        try:
            assert resize in ['expand', 'contract'], 'Incorrect resize argument: {}'.format(resize)

            start = change_basis(mx, startBasis, stdBasis1, dimOrBlockDims)
            mid = resize_std_mx(start, resize, stdBasis1, stdBasis2)
            # No else case needed, see assert
            return change_basis(mid, stdBasis2, endBasis, dimOrBlockDims)
        except ValueError:
            print('startBasis: {}'.format(startBasis))
            print('endBasis: {}'.format(endBasis))
            print('stdBasis1: {}'.format(stdBasis1))
            print(stdBasis1.dim)
            print('stdBasis2: {}'.format(stdBasis2))
            print(stdBasis2.dim)
            raise
    else:
        return change_basis(mx, startBasis, endBasis, dimOrBlockDims)

from ..tools.basisconstructors import *

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
            def is_integer(x):
                return bool( abs(x - round(x)) < 1e-6 )
            if isinstance(dimOrBlockDims, _numbers.Integral):
                dimOrBlockDims = [dimOrBlockDims]
            assert isinstance(dimOrBlockDims, _collections.Container)
            for i, dim in enumerate(dimOrBlockDims):
                nQubits = _np.log2(dim)
                if not is_integer(nQubits):
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

def state_to_pauli_density_vec(state_vec):
    """
    Convert a single qubit state vector into a density matrix.

    Parameters
    ----------
    state_vec : list or tuple
       State vector in the sigma-z basis, len(state_vec) == 2

    Returns
    -------
    numpy array
        The 2x2 density matrix of the pure state given by state_vec, given
        as a 4x1 column vector in the Pauli basis.
    """
    assert( len(state_vec) == 2 )
    st_vec = _np.array( [ [state_vec[0]], [state_vec[1]] ] )
    dm_mx = _np.kron( _np.conjugate(_np.transpose(st_vec)), st_vec ) #density matrix in sigmaz basis
    return stdmx_to_ppvec(dm_mx)

def vec_to_stdmx(v, basis, keep_complex=False):
    """
    Convert a vector in this basis to
     a matrix in the standard basis.

    Parameters
    ----------
    v : numpy array
        The vector length 4 or 16 respectively.

    Returns
    -------
    numpy array
        The matrix, 2x2 or 4x4 depending on nqubits 
    """
    dim   = int(_np.sqrt( len(v) )) # len(v) = dim^2, where dim is matrix dimension of Pauli-prod mxs
    basis = Basis(basis, dim)
    ret = _np.zeros( (dim, dim), 'complex' )
    for i, mx in enumerate(basis._matrices):
        if keep_complex:
            ret += v[i]*mx
        else:
            ret += float(v[i])*mx
    return ret

gmvec_to_stdmx  = partial(vec_to_stdmx, basis='gm')
ppvec_to_stdmx  = partial(vec_to_stdmx, basis='pp')
qtvec_to_stdmx  = partial(vec_to_stdmx, basis='qt')
stdvec_to_stdmx = partial(vec_to_stdmx, basis='std')

def stdmx_to_vec(m, basis):
    """
    Convert a matrix in the standard basis to
     a vector in the Pauli basis.

    Parameters
    ----------
    m : numpy array
        The matrix, shape 2x2 (1Q) or 4x4 (2Q)

    Returns
    -------
    numpy array
        The vector, length 4 or 16 respectively.
    """

    assert(len(m.shape) == 2 and m.shape[0] == m.shape[1])
    dim = m.shape[0]
    basis = Basis(basis, dim)
    v = _np.empty((dim**2,1))
    for i, mx in enumerate(basis._matrices):
        if basis.real:
            v[i,0] = _np.real(_mt.trace(_np.dot(mx,m)))
        else:
            v[i,0] = _mt.trace(_np.dot(mx,m))
    return v

stdmx_to_ppvec  = partial(stdmx_to_vec, basis='pp')
stdmx_to_gmvec  = partial(stdmx_to_vec, basis='gm')
stdmx_to_stdvec = partial(stdmx_to_vec, basis='std')

