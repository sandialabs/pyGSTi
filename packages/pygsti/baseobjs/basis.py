""" Defines the Basis object and supporting functions """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

from functools    import partial
from functools   import wraps
from itertools    import product, chain

import copy as _copy
import numbers as _numbers
import collections as _collections

from numpy.linalg import inv as _inv
import numpy as _np
import scipy.sparse as _sps
import scipy.sparse.linalg as _spsl

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
        in the standard basis operation matrices can have complex elements but these elements are additionally
        constrained.  This makes operation matrix parameterization and optimization much more convenient
        in the "gm" or "pp" bases.
    '''
    DefaultInfo = dict()
    CustomCount = 0 # The number of custom bases, used for serialized naming

    def __init__(self, name=None, dim=None, matrices=None, longname=None, real=None, labels=None, sparse=False):
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

        sparse : bool, optional
            Whether the basis matrices should be stored as SciPy CSR sparse matrices
            or dense numpy arrays (the default).
        '''

        self.name = None
        self.dim    = Dim([])
        self.sparse = sparse
        self._blockMatrices = None  # means "needs to be computed"
        self._matrices = None       # means "needs to be computed"
        self.longname = None
        self._labels = labels       # None means "needs to be computed"
        self.real = real


        #Set self.name, self.dim, self.sparse, self._blockMatrices, self._matrices
        if matrices is None: # no explicit matrices given - use name and dim only
            if isinstance(name, Basis): # then just copy basis
                basis = name
                self.name = basis.name
                self.dim = basis.dim
                self.sparse = basis.sparse
                self._blockMatrices = _copy.deepcopy(basis._blockMatrices) # will work for 'None' also
                self._matrices = _copy.deepcopy(basis._matrices) # will work for 'None' also
                self.longname = basis.longname
                self._labels = basis.labels
                self.real = basis.real
                
            elif isinstance(name, list): # list of Basis objs or (name,dim) pairs
                basis_list = name
                if len(basis_list) == 0: #special case of empty list
                    self._blockMatrices = [] # computed, but there aren't any!
                    self._matrices = []      # computed, but there aren't any!
                    self._labels = []         # computed, but there aren't any!
                    self.name     = "*Empty*"
                    self.longname = "Empty (0-element) basis"
                else:
                    basis = _build_composite_basis(basis_list, sparse)
                    self.__dict__.update(basis.__dict__)
            elif name is not None: # assume name is a string and build by name and dim, ie Basis('pp', 4)
                self.name   = str(name)
                self.dim    = Dim(dim) if (dim is not None) else Dim([])
                #leave _blockMatrices, _matrices, and possiblly labels as LAZY (None)
            else: # if name and matrices are none -> "Empty" basis
                self._blockMatrices = [] # computed, but there aren't any!
                self._matrices = []      # computed, but there aren't any!
                self._labels = []         # computed, but there aren't any!
                self.name     = "*Empty*"
                self.longname = "Empty (0-element) basis"
                
        else: #explicit 'matrices' given, so populate using these
            #Assume name is just a string (not lists or Basis objs, etc?)
            if name is None:
                self.name = 'CustomBasis_{}'.format(Basis.CustomCount)
                Basis.CustomCount += 1
            self.name = name
            self.longname = longname
            
            if len(matrices) > 0:
                first = matrices[0]
                if isinstance(first, tuple) or \
                        isinstance(first, Basis):
                    basis = _build_composite_basis(matrices, sparse) # really list of Bases or basis tuples
                    blockMatrices = basis.block_matrices
                elif isinstance(first, list) or \
                     (isinstance(first, _np.ndarray) and first.ndim == 3): # els of matrices are sub-bases, so
                    blockMatrices = [mxs for mxs in matrices]              
                elif (isinstance(first, _np.ndarray) or _sps.issparse(first)) \
                     and first.ndim ==2:          # matrices is a list of matrices 
                    blockMatrices = [matrices]    # so set as the first (& only) sub-basis-block
                else:
                    raise ValueError("Unknown `matrices` format: %s" % str(matrices))
            else:
                blockMatrices = []
                
            self._blockMatrices = blockMatrices
            self._matrices = _build_composite_matrices(blockMatrices, sparse=sparse)

            #Set matrices to read-only
            if not self.sparse: # sparse matrices don't have a writeable flag
                for block in self._blockMatrices:
                    for mx in block:
                        mx.flags.writeable = False
                for mx in self._matrices:
                    mx.flags.writeable = False

            #Set self.dim and check against matrix shapes:
            blockDims = [ (group[0].shape[0] if len(group)>0 else 0)
                          for group in self._blockMatrices]
            if dim is not None: #then check blockDims against given dim
                if isinstance(dim,_numbers.Integral): dims = [dim]*len(blockDims) # for len=0 case
                elif isinstance(dim,Dim): dims = dim.blockDims
                else: dims = dim # assume dim is a list of ints
                assert(list(dims) == blockDims), \
                    "Dimension mismatch in basis construction: %s != %s" % (str(dims),str(blockDims))
            self.dim = Dim(blockDims)

        if labels is not None: # if None, compute labels later (only if neede)
            if len(labels) == len(self): # len(self) gives num matrices if available, otherwise d2
                self._labels = tuple(labels)
            else:
                raise ValueError("Basis initialization error: expected a list of %d labels but got: %s"
                                 % (len(self), str(labels)))
            
        #Set self.real, self.longname w/defaults if they haven't been set yet
        def get_info(attr, default):
            """ Shorthand for retrieving a default value from the
                _basisConstructorDict dict """
            try:
                return getattr(_basisConstructorDict[self.name], attr)
            except KeyError:
                return default

        if self.real is None:
            self.real = get_info('real', default=True)
        if self.longname is None:
            self.longname = get_info('longname', default=self.name)            

    def _lazy_build_matrices_and_labels(self):
        #LAZY building of matrices (in case we never need them)
        if self._blockMatrices is None or self._matrices is None:
            self._blockMatrices = _build_default_block_matrices(self.name, self.dim, self.sparse) 
            self._matrices = _build_composite_matrices(self._blockMatrices, sparse=self.sparse)
        try:
            self._labels = basis_element_labels(self.name, self.dim.blockDims)
        except NotImplementedError:
            self._labels = []
            for i, block in enumerate(self._blockMatrices):
                for j in range(len(block)):
                    self._labels.append('M({})[{}]'.format(self.name,'{},{}'.format(i, j)))

    @property
    def block_matrices(self):
        if self._blockMatrices is None:
            self._lazy_build_matrices_and_labels()
        return self._blockMatrices

    @property
    def matrices(self):
        if self._matrices is None:
            self._lazy_build_matrices_and_labels()
        return self._matrices

    @property
    def labels(self):
        if self._labels is None:
            self._lazy_build_matrices_and_labels()
        return self._labels

    
    def copy(self):
        """Make a copy of this Basis object."""
        return Basis(self)

    def __str__(self):
        if self._labels is None: labelstr = "(no labels computed yet)"
        else: labelstr = ', '.join(self._labels)
        return '{} Basis : {}'.format(self.longname, labelstr)

    def __getitem__(self, index):
        return self.matrices[index]

    def __len__(self):
        if self._matrices is not None:
            return len(self.matrices) # if we have actual matrices, we may have a *subset* of a basis
        else: 
            return sum([ bd**2 for bd in self.dim.blockDims])

    def __eq__(self, other):
        if self.sparse and self.dim.opDim > 16:
            return False # to expensive to compare sparse matrices
        
        otherIsBasis = isinstance(other, Basis)
        if otherIsBasis and (self.sparse != other.sparse):
            return False
        
        if self.sparse:
            def sparse_equal(A,B,atol = 1e-8): 
                """ NOTE: same as matrixtools.sparse_equal - but can't import that here """
                if _np.array_equal(A.shape, B.shape)==0:
                    return False
    
                r1,c1 = A.nonzero()
                r2,c2 = B.nonzero()
    
                lidx1 = _np.ravel_multi_index((r1,c1), A.shape)
                lidx2 = _np.ravel_multi_index((r2,c2), B.shape)
                sidx1 = lidx1.argsort()
                sidx2 = lidx2.argsort()
    
                index_match = _np.array_equal(lidx1[sidx1], lidx2[sidx2])
                if index_match==0:
                    return False
                else:  
                    v1 = A.data
                    v2 = B.data        
                    V1 = v1[sidx1]
                    V2 = v2[sidx2]        
                return _np.allclose(V1,V2, atol=atol)

            if otherIsBasis:
                return all([ sparse_equal(A,B) for A,B in zip(self.matrices, other.matrices)])
            else:
                return all([ sparse_equal(A,B) for A,B in zip(self.matrices, other)])
        else:
            if otherIsBasis:
                return _np.array_equal(self.matrices, other.matrices)
            else:
                return _np.array_equal(self.matrices, other)


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
        #Note: construct to_basis as sparse this basis is sparse and
        # if to_basis is not already a Basis object
        to_basis = Basis(to_basis, self.dim.blockDims, sparse=self.sparse)

        #Note same logic as matrixtools.safedot(...)
        if to_basis.sparse: 
            return to_basis.get_from_std().dot(self.get_to_std())
        elif self.sparse:
            #return _sps.csr_matrix(to_basis.get_from_std()).dot(self.get_to_std())
            return _np.dot(to_basis.get_from_std(), self.get_to_std().toarray())
        else:
            return  _np.dot(to_basis.get_from_std(), self.get_to_std())            

        
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
        return self.block_matrices[index]

    @cache_by_hashed_args
    def is_normalized(self):
        '''
        Check if a basis is normalized

        Returns
        -------
        bool
        '''
        for i,mx in enumerate(self.matrices):
            t = _np.trace(_np.dot(mx, mx))
            t = _np.real(t)
            if not _np.isclose(t,1.0): return False
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
        numpy array or list of SciPy CSR matrices
            For a dense basis (`basis.sparse == False`), an array of matrices,
            shape == (nMatrices, d, d) where d is the composite matrix
            dimension.  For a sparse basis, a list of SciPy CSR matrices.
        '''
        return self.matrices

    
    @cache_by_hashed_args
    def get_expand_mx(self):
        '''
        Retrieve the matrix that will convert from the direct sum space to the embedding space

        Returns
        -------
        numpy array
        '''
        # Dim: dmDim 5 opDim 5 blockDims [1, 1, 1, 1, 1] embedDim 25
        assert(not self.sparse), "get_expand_mx not implemented for sparse mode"
        x = sum(len(mxs) for mxs in self.block_matrices)
        y = sum(mxs[0].shape[0] for mxs in self.block_matrices) ** 2
        expandMx = _np.zeros((x, y), 'complex')
        for i, compMx in enumerate(self.matrices):
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
        if self.sparse:
            toStd = _sps.lil_matrix((self.dim.opDim, self.dim.opDim), dtype='complex' )
        else:
            toStd = _np.zeros((self.dim.opDim, self.dim.opDim), 'complex' )
            
        #Since a multi-block basis is just the direct sum of the individual block bases,
        # transform mx is just the transfrom matrices of the individual blocks along the
        # diagonal of the total basis transform matrix

        start = 0
        for mxs in self.block_matrices:
            l = len(mxs)
            for j, mx in enumerate(mxs):
                if self.sparse:
                    assert(_sps.issparse(mx)), "Expected sparse basis elements!"
                    toStd[start:start+l,start+j] = mx.tolil().reshape((l,1)) #~flatten()
                else:
                    toStd[start:start+l,start+j] = mx.flatten()
            start += l 
        assert(start == self.dim.opDim)
        if self.sparse: toStd = toStd.tocsr()
        return toStd

    @cache_by_hashed_args
    def get_from_std(self):
        '''
        Retrieve the matrix that will convert from the standard basis to the current basis

        Returns
        -------
        numpy array
        '''
        if self.sparse:
            return _spsl.inv(self.get_to_std().tocsc()).tocsr()
        else:
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
        return Basis(otherName, self.dim.blockDims, sparse=self.sparse)

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
        return Basis(otherName, sum(self.dim.blockDims), sparse=self.sparse)

    def std_equivalent(self):
        """ Convenience method identical to `.equivalent('std')` """
        return self.equivalent('std')

    def expanded_std_equivalent(self):
        """ Convenience method identical to `.expanded_equivalent('std')` """
        return self.expanded_equivalent('std')

def _build_composite_matrices(block_matrices, sparse=False):
    '''
    Build the large composite matrices of a composite basis
    ie for std basis with dim [2, 1], build
    [[1 0 0]  [[0 1 0]  [[0 0 0]  [[0 0 0]  [[0 0 0]
     [0 0 0]   [0 0 0]   [1 0 0]   [0 1 0]   [0 0 0]
     [0 0 0]], [0 0 0]], [0 0 0]], [0 0 0]], [0 0 1]]
    For a non composite basis, this just returns the basis matrices

    Parameters
    ----------
    block_matrices : list
        A list of "blocks", where each block is a list of matrices.

    sparse : bool, optional
        Whether the created compositve matrices should be sparse or not

    Returns
    -------
    numpy array or list of SciPy CSR matrices
        For a dense basis (`basis.sparse == False`), an array of matrices,
        shape == (nMatrices, d, d) where d is the composite matrix
        dimension.  For a sparse basis, a list of SciPy CSR matrices.
    '''
    nMxs = sum([len(mxs) for mxs in block_matrices])
    length  = sum(mxs[0].shape[0] for mxs in block_matrices)
    if sparse:
        compMxs = []
    else:
        compMxs = _np.zeros( (nMxs, length, length), 'complex')
    i, start   = 0, 0

    for mxs in block_matrices:
        d = mxs[0].shape[0]
        for mx in mxs:
            assert(_sps.issparse(mx) == sparse),"Inconsistent sparsity!"
            if sparse:
                diagBlks = []
                if start > 0:
                    diagBlks.append( _sps.csr_matrix((start,start),dtype='complex') ) #zeros
                diagBlks.append(mx)
                if start+d < length:
                    diagBlks.append( _sps.csr_matrix((length-(start+d),length-(start+d)),dtype='complex') ) #zeros
                compMxs.append( _sps.block_diag(diagBlks, "csr", 'complex') )
            else:
                compMxs[i][start:start+d,start:start+d] = mx
            i += 1
        start += d 
    assert(start == length and i == nMxs)
    return compMxs

    
def _build_composite_basis(bases, sparse=False):
    '''
    Build a composite basis from a list of `(name,dim)` tuples or Basis objects
      (or a list of mixed tuples and Basis objects)

    Parameters
    ----------
    bases : list of tuples/Basis objects

    sparse : bool, optional

    Returns
    -------
    Basis
        the composite basis created
    '''
    assert len(bases) > 0, 'Need at least one basis-dim pair to compose'
    basisObjs = []
    for item in bases:
        if isinstance(item, tuple):
            basisObjs.append(Basis(name=item[0], dim=item[1], sparse=sparse))
        else:
            basisObjs.append(item)

    blockMatrices = [basis._matrices        for basis in basisObjs]
    name          = ','.join(basis.name     for basis in basisObjs)
    longname      = ','.join(basis.longname for basis in basisObjs)
    real          = all(basis.real          for basis in basisObjs)
    blockDims     = list(chain(*[ basis.dim.blockDims for basis in basisObjs]))
    sparseFlags  = [basis.sparse for basis in basisObjs]
    assert(all([s == sparseFlags[0] for s in sparseFlags])), \
        "All basis components must have same sparse flag"

    names = [basis.name for basis in basisObjs]
    if len(set(names)) == 1:
        name = names[0] #if all names are the same, retain the same name for the composite basis
    else:
        name = ','.join(names)

    if any([(mxblk is None) for mxblk in blockMatrices]):
        blockMatrices = None # if any block of basis hasn't computed it's matrices,
                             # don't initialize a Basis with explicit matrices.
    
    composite = Basis(matrices=blockMatrices, dim=Dim(blockDims), name=name, longname=longname, real=real,
                      sparse=sparseFlags[0])
    return composite


# Allow flexible basis building without cluttering the basis __init__ method with instance checking
def _build_block_matrices(name=None, dim=None, matrices=None, sparse=False):
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

    sparse : bool, optional
        Whether any built matrices should be SciPy CSR sparse matrices
        or dense numpy arrays (the default).


    Returns
    -------
    name : str
    blockMatrices : list of lists of numpy arrays
    sparse : bool
    '''
    if isinstance(name, Basis):
        basis         = name
        blockMatrices = _copy.deepcopy(basis._blockMatrices)
        name          = basis.name
        sparse        = basis.sparse
    elif isinstance(name, list):
        if len(name) == 0: #special case of empty list
            blockMatrices = []
            name          = "*Empty*"
            sparse        = sparse
        else:
            basis = _build_composite_basis(name, sparse)
            blockMatrices = basis._blockMatrices
            name          = basis.name
            sparse        = basis.sparse
    else:
        if matrices is None: # built by name and dim, ie Basis('pp', 4)            
            if name is not None: 
                matrices = _build_default_block_matrices(name, dim, sparse)
            else: # if name and matrices are none -> "Empty" basis
                name = "*Empty*"
                matrices = []

        if len(matrices) > 0:
            first = matrices[0]
            if isinstance(first, tuple) or \
                    isinstance(first, Basis):
                basis = _build_composite_basis(matrices, sparse) # really list of Bases or basis tuples
                blockMatrices = basis._blockMatrices
                name          = basis.name
            elif isinstance(first, list) or \
                 (isinstance(first, _np.ndarray) and first.ndim == 3): # els of matrices are sub-bases, so
                blockMatrices = matrices                              # set directly equal to blockMatrices
            elif (isinstance(first, _np.ndarray) or _sps.issparse(first)) \
                 and first.ndim ==2:          # matrices is a list of matrices 
                blockMatrices = [matrices]    # so set as the first (& only) sub-basis-block
        else:
            blockMatrices = []
        if name is None:
            name = 'CustomBasis_{}'.format(Basis.CustomCount)
            Basis.CustomCount += 1
    return name, blockMatrices, sparse

def _build_default_block_matrices(name, dim, sparse=False):
    '''
    Build the default block matrices for a basis object 
    (i.e. std, pp, gm, or qt basis matrices at time of writing)

    Parameters
    ----------
    name : string
        Name of the basis to be created

    dim : int
        dimension of the basis to be created.

    sparse : bool, optional
        Whether to create sparse or dense matrices

    Returns
    -------
    list of lists of numpy arrays (or SciPy sparse matrices)
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
        subBasisMxs = f(blockDim) # a list of (dense) mxs
        if not sparse:
            blockMatrices.append(subBasisMxs)
        else:
            blockMatrices.append([_sps.csr_matrix(M) for M in subBasisMxs])
    return blockMatrices


def basis_matrices(nameOrBasis, dim, sparse=False):
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

    sparse : bool, optional
        Whether any built matrices should be SciPy CSR sparse matrices
        or dense numpy arrays (the default).

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
        if len(basis) == 0: return [] # special case of empty basis - don't check dim in this case
        assert(basis.dim.dmDim == dim), "Basis object has wrong dimension ({}) for requested basis matrices ({})".format(
            basis.dim.dmDim, dim)
        return basis.matrices

    name = nameOrBasis
    if name not in _basisConstructorDict:
        raise NotImplementedError('No instructions to create supposed \'default\' basis:  {} of dim {}'.format(
            name, dim))
    f = _basisConstructorDict[name].constructor
    if not sparse:
        return f(dim)
    else:
        return [_sps.csr_matrix(M) for M in f(dim)]

    
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


def basis_element_labels(basis, dimOrBlockDims=None):
    """
    Returns a list of short labels corresponding to to the
    elements of the described basis.  These labels are
    typically used to label the rows/columns of a box-plot
    of a matrix in the basis.

    Parameters
    ----------
    basis : {'std', 'gm', 'pp', 'qt'}
        Which basis the model is represented in.  Allowed
        options are Matrix-unit (std), Gell-Mann (gm),
        Pauli-product (pp) and Qutrit (qt).  If the basis is
        not known, then an empty list is returned.

    dimOrBlockDims : int or list, optional
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

    assert(dimOrBlockDims is not None), \
        "Must specify `dimOrBlockDims` when `basis` isn't a Basis object"
    
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
