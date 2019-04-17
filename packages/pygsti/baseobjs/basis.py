""" Defines the Basis object and supporting functions """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

from functools import partial
from functools import wraps
from itertools import product, chain

import copy as _copy
import numbers as _numbers
import collections as _collections
import warnings as _warnings
import itertools as _itertools

from numpy.linalg import inv as _inv
import numpy as _np
import scipy.sparse as _sps
import scipy.sparse.linalg as _spsl
import copy as _copy

import math

from .basisconstructors import _basisConstructorDict
from .basisconstructors import cache_by_hashed_args

#Helper functions
try: basestring
except NameError: basestring = str


def _isstr(x):
    """ Same as compattools.py, but can't import that here """
    return isinstance(x, basestring)


def _sparse_equal(A, B, atol=1e-8):
    """ NOTE: same as matrixtools.sparse_equal - but can't import that here """
    if _np.array_equal(A.shape, B.shape) == 0:
        return False

    r1, c1 = A.nonzero()
    r2, c2 = B.nonzero()

    lidx1 = _np.ravel_multi_index((r1, c1), A.shape)
    lidx2 = _np.ravel_multi_index((r2, c2), B.shape)
    sidx1 = lidx1.argsort()
    sidx2 = lidx2.argsort()

    index_match = _np.array_equal(lidx1[sidx1], lidx2[sidx2])
    if index_match == 0:
        return False
    else:
        v1 = A.data
        v2 = B.data
        V1 = v1[sidx1]
        V2 = v2[sidx2]
    return _np.allclose(V1, V2, atol=atol)


class Basis(object):
    '''
    TODO: docstrings in this module
    Base class = abstract notion of a basis with a certain size - no actual elements or labels
     are stored though it *expects* to have .elements and .labels members to perform common
     basis operations.

    Encapsulates a basis - an ordered set of labelled matrices/vectors.
    '''

    @classmethod
    def cast(cls, nameOrBasisOrMatrices, dim=None, sparse=None, classicalName='cl'):
        #print("DB: CAST = ",nameOrBasisOrMatrices,dim)
        from ..objects.labeldicts import StateSpaceLabels as _SSLs
        if nameOrBasisOrMatrices is None:  # special case of empty basis
            return ExplicitBasis([], [], "*Empty*", "Empty (0-element) basis", False, sparse)  # empty basis
        elif isinstance(nameOrBasisOrMatrices, Basis):
            #then just check to make sure consistent with `dim` & `sparse`
            basis = nameOrBasisOrMatrices
            if dim is not None:
                assert(dim == basis.dim), "Basis object has unexpected dimension: %d vs %d" % (dim, basis.dim)
            if sparse is not None:
                assert(sparse == basis.sparse), "Basis object has unexpected sparsity: %s" % (basis.sparse)
            return basis
        elif _isstr(nameOrBasisOrMatrices):
            name = nameOrBasisOrMatrices
            if isinstance(dim, _SSLs):
                sslbls = dim
                tpbBases = []
                for tpbLabels in sslbls.labels:
                    if len(tpbLabels) == 1:
                        nm = name if (sslbls.labeltypes[tpbLabels[0]] == 'Q') else classicalName
                        tpbBases.append(BuiltinBasis(nm, sslbls.labeldims[tpbLabels[0]], sparse))
                    else:
                        tpbBases.append(TensorProdBasis([
                            BuiltinBasis(name if (sslbls.labeltypes[l] == 'Q') else classicalName,
                                         sslbls.labeldims[l], sparse) for l in tpbLabels]))
                if len(tpbBases) == 1:
                    return tpbBases[0]
                else:
                    return DirectSumBasis(tpbBases)
            elif isinstance(dim, (list, tuple)):  # list/tuple of block dimensions
                tpbBases = []
                for tpbDim in dim:
                    if isinstance(tpbDim, (list, tuple)):  # list/tuple of tensor-product dimensions
                        tpbBases.append(
                            TensorProdBasis([BuiltinBasis(name, factorDim, sparse) for factorDim in tpbDim]))
                    else:
                        tpbBases.append(BuiltinBasis(name, tpbDim, sparse))
                return DirectSumBasis(tpbBases)
            else:
                return BuiltinBasis(name, dim, sparse)
        elif isinstance(nameOrBasisOrMatrices, (list, tuple, _np.ndarray)):
            # assume a list/array of matrices or (name, dim) pairs
            if len(nameOrBasisOrMatrices) == 0:  # special case of empty basis
                return ExplicitBasis([], [], "*Empty*", "Empty (0-element) basis", False, sparse)  # empty basis
            elif isinstance(nameOrBasisOrMatrices[0], _np.ndarray):
                b = ExplicitBasis(nameOrBasisOrMatrices, sparse=sparse)
                if dim is not None:
                    assert(dim == b.dim), "Created explicit basis has unexpected dimension: %d vs %d" % (dim, b.dim)
                if sparse is not None:
                    assert(sparse == b.sparse), "Basis object has unexpected sparsity: %s" % (b.sparse)
                return b
            else:  # assume els are (name, dim) pairs
                compBases = [BuiltinBasis(subname, subdim, sparse)
                             for (subname, subdim) in nameOrBasisOrMatrices]
                return DirectSumBasis(compBases)

        else:
            raise ValueError("Can't cast %s to be a basis!" % str(type(nameOrBasisOrMatrices)))

    def __init__(self, name, longname, dim, size, elshape, real, sparse):
        self.name = name
        self.longname = longname
        self.dim = dim  # dimension of vector space this basis fully or partially spans
        self.size = size  # number of elements (== dim if a *full* basis)
        self.elshape = tuple(elshape)  # shape of "natural" elements - size may be > self.dim (to display naturally)
        # whether coefficients must be real (*not* whether elements are real - they're always complex)
        self.real = real
        self.sparse = sparse  # whether elements are stored as sparse vectors/matrices

    @property
    def elndim(self):
        if self.elshape is None: return 0
        return len(self.elshape)

    @property
    def elsize(self):
        if self.elshape is None: return 0
        return int(_np.product(self.elshape))

    def is_simple(self):
        """
        Whether the flattened-element vector space is the *same*
        space as the space this basis belongs to"""
        return self.elsize == self.dim

    def is_complete(self):
        """
        Whether this is a complete basis, i.e. this basis's
        vectors span the entire space that they live in.
        """
        return self.dim == self.size

    def is_partial(self):
        return not self.is_complete()

    @property
    def vector_elements(self):
        if self.sparse:
            return [_sps.lil_matrix(el).reshape((self.elsize, 1)) for el in self.elements]
        else:
            return [el.flatten() for el in self.elements]

    def copy(self):
        """Make a copy of this Basis object."""
        return _copy.deepcopy(self)

    def __str__(self):
        return '{} (dim={}), {} elements of shape {} :\n{}'.format(
            self.longname, self.dim, self.size, self.elshape, ', '.join(self.labels))

    def __getitem__(self, index):
        return self.elements[index]

    def __len__(self):
        return self.size

    def __eq__(self, other):
        otherIsBasis = isinstance(other, Basis)

        if otherIsBasis and (self.sparse != other.sparse):  # sparseness mismatch => not equal
            return False

        if self.sparse:
            if self.dim > 256:
                _warnings.warn("Attempted comparison between bases with large sparse matrices!  Assuming not equal.")
                return False  # to expensive to compare sparse matrices

            if otherIsBasis:
                return all([_sparse_equal(A, B) for A, B in zip(self.elements, other.elements)])
            else:
                return all([_sparse_equal(A, B) for A, B in zip(self.elements, other)])
        else:
            if otherIsBasis:
                return _np.array_equal(self.elements, other.elements)
            else:
                return _np.array_equal(self.elements, other)

    def transform_matrix(self, to_basis):
        '''
        TODO: docstring
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
        if not isinstance(to_basis, Basis):
            to_basis = self.equivalent(to_basis)

        #Note same logic as matrixtools.safedot(...)
        if to_basis.sparse:
            return to_basis.get_from_std().dot(self.get_to_std())
        elif self.sparse:
            #return _sps.csr_matrix(to_basis.get_from_std()).dot(self.get_to_std())
            return _np.dot(to_basis.get_from_std(), self.get_to_std().toarray())
        else:
            return _np.dot(to_basis.get_from_std(), self.get_to_std())

    def reverse_transform_matrix(self, from_basis):
        '''
        TODO: docstring (and above) - the matrix to transform *to* this basis from `from_basis`
        '''
        if not isinstance(from_basis, Basis):
            from_basis = self.equivalent(from_basis)

        #Note same logic as matrixtools.safedot(...)
        if self.sparse:
            return self.get_from_std().dot(from_basis.get_to_std())
        elif from_basis.sparse:
            #return _sps.csr_matrix(to_basis.get_from_std()).dot(self.get_to_std())
            return _np.dot(self.get_from_std(), from_basis.get_to_std().toarray())
        else:
            return _np.dot(self.get_from_std(), from_basis.get_to_std())

    @cache_by_hashed_args
    def is_normalized(self):
        '''
        Check if a basis is normalized

        Returns
        -------
        bool
        '''
        if self.elndim == 2:
            for i, mx in enumerate(self.elements):
                t = _np.trace(_np.dot(mx, mx))
                t = _np.real(t)
                if not _np.isclose(t, 1.0): return False
            return True
        elif self.elndim == 1:
            raise NotImplementedError("TODO: add code so this works for *vector*-valued bases too!")
        else:
            raise ValueError("I don't know what normalized means for elndim == %d!" % self.elndim)

    @cache_by_hashed_args
    def get_to_std(self):
        '''
        Retrieve the matrix that will convert from the current basis to the standard basis

        Returns
        -------
        numpy array
        '''
        if self.sparse:
            toStd = _sps.lil_matrix((self.dim, self.size), dtype='complex')
        else:
            toStd = _np.zeros((self.dim, self.size), 'complex')

        for i, vel in enumerate(self.vector_elements):
            toStd[:, i] = vel
        return toStd

    @cache_by_hashed_args
    def get_from_std(self):
        '''
        Retrieve the matrix that will convert from the standard basis to the current basis

        Returns
        -------
        numpy array
        '''
        #TODO: warning/error/pseudoinverse when get_to_std() returns a non-square matrix?
        if self.sparse:
            return _spsl.inv(self.get_to_std().tocsc()).tocsr()
        else:
            return _inv(self.get_to_std())

    @cache_by_hashed_args
    def get_to_simple_std(self):  # OLD: contract_mx(self):
        '''
        Retrieve the matrix that will convert from the embedding space to the direct sum space,
        truncating if necessary (Currently without warning)
        TODO: docstring

        Returns
        -------
        numpy array
        '''
        # Default implementation assumes that the (flattened) element space
        # *is* a standard representation of the vector space this basis or partial-basis
        # acts upon (this is *not* true for direct-sum bases, where the flattened
        # elements represent vectors in the *embedding* space (w/larger dim than actual space).
        return self.get_to_std()  # because this *is* a simple basis

    @cache_by_hashed_args
    def get_from_simple_std(self):  # OLD: get_expand_mx(self):
        '''
        Retrieve the matrix that will convert from the direct sum space to the embedding space
        TODO: docstring

        Returns
        -------
        numpy array
        '''
        # Default implementation assumes that the (flattened) element space
        # *is* a standard representation of the vector space this basis or partial-basis
        # acts upon (this is *not* true for direct-sum bases, where the flattened
        # elements represent vectors in the *embedding* space (w/larger dim than actual space).
        assert(not self.sparse), "get_from_simple_std not implemented for sparse mode"  # (b/c pinv used)
        return _np.linalg.pinv(self.get_to_simple_std())

    def equivalent(self, builtinBasisName):
        """
        Create a Basis that is equivalent in structure & dimension to this
        basis but whose simple components (perhaps just this basis) is of
        the builtin basis type given by `builtinBasisName`.
        """
        return BuiltinBasis(builtinBasisName, self.dim, sparse=self.sparse)

        #"""
        #Return a `Basis` of the type given by `otherName` and the dimensions
        #of this `Basis`.
        #
        #Parameters
        #----------
        #otherName : {'std', 'gm', 'pp', 'qt'}
        #    A standard basis abbreviation.
        #
        #Returns
        #-------
        #Basis
        #"""

    def simple_equivalent(self, builtinBasisName=None):
        """
        Create a simple basis (i.e. one without components) of the builtin
        type specified whose dimension is compatible with the *elements* of
        this basis.  (e.g. the embedded space for a direct-sum basis)
        """
        if builtinBasisName is None: return self.copy()
        else: return self.equivalent(builtinBasisName)

        #"""
        #Return a single-block `Basis` of the type given by `otherName` and
        #dimension given by the sum of the block dimensions of this `Basis`.
        #
        #Parameters
        #----------
        #otherName : {'std', 'gm', 'pp', 'qt', None}
        #    A standard basis abbreviation.  If None, then this
        #    `Basis`'s name is used.
        #
        #Returns
        #-------
        #Basis
        #"""

    #TODO REMOVE
    #def std_equivalent(self):
    #    """ Convenience method identical to `.equivalent('std')` """
    #    return self.equivalent('std')
    #
    #def simple_std_equivalent(self):
    #    """ Convenience method identical to `.simple_equivalent('std')` """
    #    return self.simple_equivalent('std')


class LazyBasis(Basis):
    """
    TODO: docstring - basis with elements that are constructed on an as-needed basis (no pun intended)
    """

    def __init__(self, name, longname, dim, size, elshape, real, sparse):
        self._elements = None        # "natural-shape" elements - can be vecs or matrices
        self._labels = None          # element labels
        super(LazyBasis, self).__init__(name, longname, dim, size, elshape, real, sparse)

    def _lazy_build_elements(self):
        raise NotImplementedError("Derived classes must implement this function!")

    def _lazy_build_labels(self):
        raise NotImplementedError("Derived classes must implement this function!")

    @property
    def elements(self):
        if self._elements is None:
            self._lazy_build_elements()
        return self._elements

    @property
    def labels(self):
        if self._labels is None:
            self._lazy_build_labels()
        return self._labels

    def __str__(self):
        if self._labels is None and self.dim > 64:
            labelstr = "(no labels computed yet)"
            return '{} (dim={}), {} elements of shape {} (not computed yet)'.format(
                self.longname, self.dim, self.size, self.elshape)
        else:
            return super(LazyBasis, self).__str__()


class ExplicitBasis(Basis):
    """ Supply elements directly """
    Count = 0  # The number of custom bases, used for serialized naming

    def __init__(self, elements, labels=None, name=None, longname=None, real=False, sparse=None):
        '''
            TODO: docstring
        '''
        if name is None:
            name = 'ExplicitBasis_{}'.format(ExplicitBasis.Count)
            if longname is None: longname = "Auto-named " + name
            ExplicitBasis.Count += 1
        elif longname is None: longname = name

        if labels is None: labels = ["E%d" % k for k in range(len(elements))]
        if (len(labels) != len(elements)):
            raise ValueError("Expected a list of %d labels but got: %s" % (len(elements), str(labels)))

        self.labels = labels
        self.elements = []
        size = len(elements)
        if size == 0:
            elshape = (); dim = 0; sparse = False
        else:
            if sparse is None:
                sparse = _sps.issparse(elements[0]) if len(elements) > 0 else False
            elshape = None
            for el in elements:
                if sparse:
                    if not _sps.issparse(el):
                        el = _sps.csr_matrix(el)  # try to convert to a sparse matrix
                else:
                    if not isinstance(el, _np.ndarray):
                        el = _np.array(el)  # try to convert to a numpy array

                if elshape is None: elshape = el.shape
                else: assert(elshape == el.shape), "Inconsistent element shapes!"
                self.elements.append(el)
            dim = int(_np.product(elshape))

        super(ExplicitBasis, self).__init__(name, longname, dim, size, elshape, real, sparse)

    def __hash__(self):
        return hash((self.name, self.dim, self.elshape, self.sparse))  # better?


class BuiltinBasis(LazyBasis):

    def __init__(self, name, dim, sparse=False):
        '''
        Initialize a basis object.
        - NOTE: dim, sparse, ... are the "creation args" for this type of basis
        TODO: docstring: matrices -> elements, dim changed to cargs == vector-space dimension, not matrix!

        Parameters
        ----------
        name : string
            Name of the basis to be created;
            if the name is 'pp', 'std', 'gm', or 'qt' and a dimension is provided,
            then a default basis is created

        dim : int
            dimension of the basis to be created.
            Only required when creating default bases

        matrices : list of numpy arrays, list of lists of numpy arrays, list of Basis objects/tuples
            Flexible argument that allows different types of basis creation
            When a list of numpy arrays, creates a non composite basis
            When a list of lists of numpy arrays or list of other bases, creates a composite bases with each outer list
            element as a composite part.

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
        assert(name in _basisConstructorDict), "Unknown builtin basis name '%s'!" % name
        if sparse is None: sparse = False  # choose dense matrices by default (when sparsity is "unspecified")
        self.cargs = {'dim': dim, 'sparse': sparse}

        longname = _basisConstructorDict[name].longname
        real = _basisConstructorDict[name].real
        size, dim, elshape = _basisConstructorDict[name].sizes(**self.cargs)
        super(BuiltinBasis, self).__init__(name, longname, dim, size, elshape, real, sparse)

        #Check that sparse is True only when elements are *matrices*
        assert(not self.sparse or self.elndim == 2), "`sparse == True` is only allowed for *matrix*-valued bases!"

    def __hash__(self):
        return hash((self.name, self.dim, self.sparse))

    def _lazy_build_elements(self):
        f = _basisConstructorDict[self.name].constructor
        self._elements = _np.array(f(**self.cargs))  # a list of (dense) mxs -> ndarray (possibly sparse in future?)
        assert(len(self._elements) == self.size), "Logic error: wrong number of elements were created!"

    def _lazy_build_labels(self):
        f = _basisConstructorDict[self.name].labeler
        self._labels = f(**self.cargs)

    def __eq__(self, other):
        if isinstance(other, BuiltinBasis):  # then can compare quickly
            return (self.name == other.name) and (self.cargs == other.cargs) and (self.sparse == other.sparse)
        elif _isstr(other):
            return self.name == other  # see if other is a string equal to our name
        else:
            return LazyBasis.__eq__(self, other)


class DirectSumBasis(LazyBasis):
    '''
    A basis that is the direct sum of one or more "component" bases.  Elements
    of this basis are simply the union of the basis elements on each component.
    TODO: docstring (more?)
    '''

    def __init__(self, component_bases, name=None, longname=None):
        '''
        TODO: docstring
        '''
        assert(len(component_bases) > 0), "Must supply at least one component basis"

        self.component_bases = []
        self._vector_elements = None  # vectorized elements: 1D arrays

        for compbasis in component_bases:
            if isinstance(compbasis, Basis):
                self.component_bases.append(compbasis)
            else:
                #compbasis can be a list/tuple of args to Basis.__init__, e.g. ('pp',2)
                self.component_bases.append(Basis(*compbasis))

        if name is None:
            name = "+".join([c.name for c in self.component_bases])
        if longname is None:
            longname = "Direct-sum basis with components " + ", ".join(
                [c.name for c in self.component_bases])

        real = all([c.real for c in self.component_bases])
        sparse = all([c.sparse for c in self.component_bases])
        assert(all([c.real == real for c in self.component_bases])), "Inconsistent `real` value among component bases!"
        assert(all([c.sparse == sparse for c in self.component_bases])), "Inconsistent sparsity among component bases!"

        #Init everything but elements and labels & their number/size
        dim = sum([c.dim for c in self.component_bases])
        elndim = len(self.component_bases[0].elshape)
        assert(all([len(c.elshape) == elndim for c in self.component_bases])
               ), "Inconsistent element ndims among component bases!"
        elshape = [sum([c.elshape[k] for c in self.component_bases]) for k in range(elndim)]
        size = sum([c.size for c in self.component_bases])
        super(DirectSumBasis, self).__init__(name, longname, dim, size, elshape, real, sparse)

    def __hash__(self):
        return hash(tuple((hash(comp) for comp in self.component_bases)))

    def _lazy_build_vector_elements(self):
        if self.sparse:
            compMxs = []
        else:
            compMxs = _np.zeros((self.size, self.dim), 'complex')

        i, start = 0, 0
        for compbasis in self.component_bases:
            for lbl, vel in zip(compbasis.labels, compbasis.vector_elements):
                assert(_sps.issparse(vel) == self.sparse), "Inconsistent sparsity!"
                if self.sparse:
                    mx = _sps.lil_matrix((self.dim, 1))
                    mx[start:start + compbasis.dim, 0] = vel
                    compMxs.append(mx)
                else:
                    compMxs[i, start:start + compbasis.dim] = vel
                i += 1
            start += compbasis.dim

        assert(i == self.size)
        self._vector_elements = compMxs

    def _lazy_build_elements(self):
        self._elements = []

        for vel in self.vector_elements:
            vstart = 0
            if self.sparse:  # build block-diagonal sparse mx
                diagBlks = []
                for compbasis in self.component_bases:
                    cs = compbasis.elshape
                    comp_vel = vel[vstart:vstart + compbasis.dim]
                    diagBlks.append(comp_vel.reshape(cs))
                    vstart += compbasis.dim
                el = _sps.block_diag(diagBlks, "csr", 'complex')

            else:
                start = [0] * self.elndim
                el = _np.zeros(self.elshape, 'complex')
                for compbasis in self.component_bases:
                    cs = compbasis.elshape
                    comp_vel = vel[vstart:vstart + compbasis.dim]
                    slc = tuple([slice(start[k], start[k] + cs[k]) for k in range(self.elndim)])
                    el[slc] = comp_vel.reshape(cs)
                    vstart += compbasis.dim
                    for k in range(self.elndim): start[k] += cs[k]

            self._elements.append(el)
        if not self.sparse:  # _elements should be an array rather than a list
            self._elements = _np.array(self._elements)

    def _lazy_build_labels(self):
        self._labels = []
        for i, compbasis in enumerate(self.component_bases):
            for lbl in compbasis.labels:
                self._labels.append(lbl + "/%d" % i)

    def __eq__(self, other):
        otherIsBasis = isinstance(other, DirectSumBasis)
        if not otherIsBasis: return False  # can't be equal to a non-DirectSumBasis
        if len(self.component_bases) != len(other.component_bases): return False
        return all([c1 == c2 for (c1, c2) in zip(self.component_bases, other.component_bases)])

    @property
    def vector_elements(self):
        if self._vector_elements is None:
            self._lazy_build_vector_elements()
        return self._vector_elements

    @cache_by_hashed_args
    def get_to_std(self):
        '''
        Retrieve the matrix that will convert from the current basis to the standard basis

        Returns
        -------
        numpy array
        '''
        if self.sparse:
            toStd = _sps.lil_matrix((self.dim, self.size), dtype='complex')
        else:
            toStd = _np.zeros((self.dim, self.size), 'complex')

        #use vector elements, which are not just flattened elements
        # (and are computed separately)
        for i, vel in enumerate(self.vector_elements):
            toStd[:, i] = vel
        return toStd

    @cache_by_hashed_args
    def get_to_simple_std(self):  # OLD: contract_mx(self):
        '''
        Retrieve the matrix that will convert from the embedding space to the direct sum space,
        truncating if necessary (Currently without warning)
        TODO: docstring

        Returns
        -------
        numpy array
        '''
        assert(not self.sparse), "get_to_simple_std not implemented for sparse mode"
        expanddim = self.elsize  # == _np.product(self.elshape)
        if self.sparse:
            toSimpleStd = _sps.lil_matrix((expanddim, self.size), dtype='complex')
        else:
            toSimpleStd = _np.zeros((expanddim, self.size), 'complex')

        for i, el in enumerate(self.elements):
            if self.sparse:
                vel = _sps.lil_matrix(el.reshape(-1, 1))  # sparse vector == sparse n x 1 matrix
            else:
                vel = el.flatten()
            toSimpleStd[:, i] = vel
        return toSimpleStd

    def equivalent(self, otherName):
        equiv_components = [c.equivalent(otherName) for c in self.component_bases]
        return DirectSumBasis(equiv_components)

    def simple_equivalent(self, otherName=None):
        """
        TODO: docstring
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
            otherName = self.name  # default
            if len(self.component_bases) > 0:
                first_comp_name = self.component_bases[0].name
                if all([c.name == first_comp_name for c in self.component_bases]):
                    otherName = first_comp_name  # if all components have the same name
        return BuiltinBasis(otherName, self.elsize, sparse=self.sparse)  # Note: changes dimension


class TensorProdBasis(LazyBasis):
    '''
    A basis that is the tensor product of one or more "component" bases.
    TODO: docstring (more?)
    '''

    def __init__(self, component_bases, name=None, longname=None):
        '''
        TODO: docstring
        '''
        assert(len(component_bases) > 0), "Must supply at least one component basis"

        self.component_bases = []
        for compbasis in component_bases:
            if isinstance(compbasis, Basis):
                self.component_bases.append(compbasis)
            else:
                #compbasis can be a list/tuple of args to Basis.__init__, e.g. ('pp',2)
                self.component_bases.append(Basis(*compbasis))

        if name is None:
            name = "*".join([c.name for c in self.component_bases])
        if longname is None:
            longname = "Tensor-product basis with components " + ", ".join(
                [c.name for c in self.component_bases])

        real = all([c.real for c in self.component_bases])
        sparse = all([c.sparse for c in self.component_bases])
        assert(all([c.real == real for c in self.component_bases])), "Inconsistent `real` value among component bases!"
        assert(all([c.sparse == sparse for c in self.component_bases])), "Inconsistent sparsity among component bases!"
        assert(sparse == False), "Sparse matrices are not supported within TensorProductBasis objects yet"

        dim = int(_np.product([c.dim for c in self.component_bases]))
        assert(all([c.dim == c.elsize for c in self.component_bases])), \
            "Components of a tensor product basis must have vector-dimension == size of elements"
        # because we use the natural representation to take tensor (kronecker) products.

        size = int(_np.product([c.size for c in self.component_bases]))
        elndim = max([c.elndim for c in self.component_bases])
        elshape = [1] * elndim
        for c in self.component_bases:
            off = elndim - c.elndim
            for k, d in enumerate(c.elshape):
                elshape[k + off] *= d

        super(TensorProdBasis, self).__init__(name, longname, dim, size, elshape, real, sparse)

    def __hash__(self):
        return hash(tuple((hash(comp) for comp in self.component_bases)))

    def _lazy_build_elements(self):
        #LAZY building of elements (in case we never need them)
        compMxs = _np.zeros((self.size,) + self.elshape, 'complex')

        #Take kronecker product of *natural* reps of component-basis elements
        # then reshape to vectors at the end.  This requires that the vector-
        # dimension of the component spaces equals the "natural space" dimension.
        comp_els = [c.elements for c in self.component_bases]
        for i, factors in enumerate(_itertools.product(*comp_els)):
            M = _np.identity(1, 'complex')
            for f in factors:
                M = _np.kron(M, f)
            compMxs[i] = M
        self._elements = compMxs

    def _lazy_build_labels(self):
        self._labels = []
        comp_lbls = [c.labels for c in self.component_bases]
        for i, factor_lbls in enumerate(_itertools.product(*comp_lbls)):
            self._labels.append(''.join(factor_lbls))

    def __eq__(self, other):
        otherIsBasis = isinstance(other, TensorProdBasis)
        if not otherIsBasis: return False  # can't be equal to a non-DirectSumBasis
        if len(self.component_bases) != len(other.component_bases): return False
        return all([c1 == c2 for (c1, c2) in zip(self.component_bases, other.component_bases)])

    def equivalent(self, otherName):
        equiv_components = [c.equivalent(otherName) for c in self.component_bases]
        return TensorProdBasis(equiv_components)

    def simple_equivalent(self, otherName=None):
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
            otherName = self.name  # default
            if len(self.component_bases) > 0:
                first_comp_name = self.component_bases[0].name
                if all([c.name == first_comp_name for c in self.component_bases]):
                    otherName = first_comp_name  # if all components have the same name
        return BuiltinBasis(otherName, self.elsize, sparse=self.sparse)


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
    return Basis.cast(nameOrBasis, dim, sparse).elements


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


def basis_element_labels(basis, dim):
    """
    Returns a list of short labels corresponding to to the
    elements of the described basis.  These labels are
    typically used to label the rows/columns of a box-plot
    of a matrix in the basis.
    TODO: docstring - update w/cargs

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
    return Basis.cast(basis, dim).labels
