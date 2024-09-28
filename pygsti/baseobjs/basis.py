"""
Defines the Basis object and supporting functions
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import copy as _copy
import itertools as _itertools
import warnings as _warnings
from functools import lru_cache
from typing import Union, Tuple, List

import numpy as _np
import scipy.sparse as _sps
import scipy.sparse.linalg as _spsl
from numpy.linalg import inv as _inv

from pygsti.baseobjs.basisconstructors import _basis_constructor_dict
from pygsti.baseobjs.statespace import StateSpace as _StateSpace
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable


#Helper functions
def _sparse_equal(a, b, atol=1e-8):
    """ NOTE: same as matrixtools.sparse_equal - but can't import that here """
    if _np.array_equal(a.shape, b.shape) == 0:
        return False

    r1, c1 = a.nonzero()
    r2, c2 = b.nonzero()

    lidx1 = _np.ravel_multi_index((r1, c1), a.shape)
    lidx2 = _np.ravel_multi_index((r2, c2), b.shape)
    sidx1 = lidx1.argsort()
    sidx2 = lidx2.argsort()

    index_match = _np.array_equal(lidx1[sidx1], lidx2[sidx2])
    if index_match == 0:
        return False
    else:
        v1 = a.data
        v2 = b.data
        V1 = v1[sidx1]
        V2 = v2[sidx2]
    return _np.allclose(V1, V2, atol=atol)


class Basis(_NicelySerializable):
    """
    An ordered set of labeled matrices/vectors.

    The base class for basis objects.  A basis in pyGSTi is an abstract notion
    of a set of labeled elements, or "vectors".  Each basis has a certain size,
    and has .elements, .labels, and .ellookup members, the latter being a
    dictionary mapping of labels to elements.

    An important point to note that isn't immediately
    intuitive is that while Basis object holds *elements* (in its
    `.elements` property) these are not the same as its *vectors*
    (given by the object's `vector_elements` property).  Often times,
    in what we term a "simple" basis, the you just flatten an element
    to get the corresponding vector-element.  This works for bases
    where the elements are either vectors (where flattening does
    nothing) and matrices.  By storing `elements` as distinct from
    `vector_elements`, the Basis can capture additional structure
    of the elements (such as viewing them as matrices) that can
    be helpful for their display and interpretation.  The elements
    are also sometimes referred to as the "natural elements" because
    they represent how to display the element in a natrual way.  A
    non-simple basis occurs when vector_elements need to be stored as
    elements in a larger "embedded" way so that these elements can be
    displayed and interpeted naturally.

    A second important note is that there is assumed to be some underlying
    "standard" basis underneath all the bases in pyGSTi.  The elements in
    a Basis are *always* written in this standard basis.  In the case of the
    "std"-named basis in pyGSTi, these elements are just the trivial vector
    or matrix units, so one can rightly view the "std" pyGSTi basis as the
    "standard" basis for a that particular dimension.

    The arguments below describe the basic properties of all basis
    objects in pyGSTi.  It is important to remember that the
    `vector_elements` of a basis are different from its `elements`
    (see the :class:`Basis` docstring), and that `dim` refers
    to the vector elements whereas elshape refers to the elements.

    For example, consider a 2-element Basis containing the I and X Pauli
    matrices.  The `size` of this basis is `2`, as there are two elements
    (and two vector elements).  Since vector elements are the length-4
    flattened Pauli matrices, the dimension (`dim`) is `4`.  Since the
    elements are 2x2 Pauli matrices, the `elshape` is `(2,2)`.

    As another example consider a basis which spans all the diagonal
    2x2 matrices.  The elements of this basis are the two matrix units
    with a 1 in the (0,0) or (1,1) location.  The vector elements,
    however, are the length-2 [1,0] and [0,1] vectors obtained by extracting
    just the diagonal entries from each basis element.  Thus, for this
    basis, `size=2`, `dim=2`, and `elshape=(2,2)` - so the dimension is
    not just the product of `elshape` entries (equivalently, `elsize`).

    Parameters
    ----------
    name : string
        The name of the basis.  This can be anything, but is
        usually short and abbreviated.  There are several
        types of bases built into pyGSTi that can be constructed by
        this name.

    longname : string
        A more descriptive name for the basis.

    real : bool
        Elements and vector elements are always allowed to have complex
        entries.  This argument indicates whether the coefficients in the
        expression of an arbitrary vector in this basis must be real.  For
        example, if `real=True`, then when pyGSTi transforms a vector in
        some other basis to a vector in *this* basis, it will demand that
        the values of that vector (i.e. the coefficients which multiply
        this basis's elements to obtain a vector in the "standard" basis)
        are real.

    sparse : bool
        Whether the elements of `.elements` for this Basis are stored (when
        they are stored at all) as sparse matrices or vectors.

    Attributes
    ----------
    dim : int
        The dimension of the vector space this basis fully or partially
        spans.  Equivalently, the length of the `vector_elements` of the
        basis.

    size : int
        The number of elements (or vector-elements) in the basis.

    elshape : int
        The shape of each element.  Typically either a length-1 or length-2
        tuple, corresponding to vector or matrix elements, respectively.
        Note that *vector elements* always have shape `(dim,)` (or `(dim,1)`
        in the sparse case).

    elndim : int
        The number of element dimensions, i.e. `len(self.elshape)`

    elsize : int
        The total element size, i.e. `product(self.elshape)`

    vector_elements : list
        The "vectors" of this basis, always 1D (sparse or dense) arrays.
    """

    # Implementation note: casting functions are classmethods, but current implementations
    # could be static methods.

    @classmethod
    def cast_from_name_and_statespace(cls, name: str, state_space: _StateSpace, sparse=None, classical_name='cl'):
        tpbBases = []
        if len(state_space.tensor_product_blocks_labels) == 1 \
            and len(state_space.tensor_product_blocks_labels[0]) == 1:
            #Special case when we can actually pipe state_space to the BuiltinBasis constructor
            lbl = state_space.tensor_product_blocks_labels[0][0]
            nm = name if (state_space.label_type(lbl) == 'Q') else classical_name
            tpbBases.append(BuiltinBasis(nm, state_space, sparse))
        else:
            #TODO: add methods to StateSpace that can extract a sub-*StateSpace* object for a given label.
            for tpbLabels in state_space.tensor_product_blocks_labels:
                if len(tpbLabels) == 1:
                    nm = name if (state_space.label_type(tpbLabels[0]) == 'Q') else classical_name
                    tpbBases.append(BuiltinBasis(nm, state_space.label_dimension(tpbLabels[0]), sparse))
                else:
                    tpbBases.append(TensorProdBasis([
                        BuiltinBasis(name if (state_space.label_type(l) == 'Q') else classical_name,
                                        state_space.label_dimension(l), sparse) for l in tpbLabels]))
        if len(tpbBases) == 1:
            return tpbBases[0]
        else:
            return DirectSumBasis(tpbBases)

    @classmethod
    def cast_from_name_and_dims(cls, name: str, dim: Union[int,list,tuple], sparse=None):
        if isinstance(dim, (list, tuple)):  # list/tuple of block dimensions
            tpbBases = []
            for tpbDim in dim:
                if isinstance(tpbDim, (list, tuple)):  # list/tuple of tensor-product dimensions
                    tpbBases.append(
                        TensorProdBasis([BuiltinBasis(name, factorDim, sparse) for factorDim in tpbDim]))
                else:
                    tpbBases.append(BuiltinBasis(name, tpbDim, sparse))

            if len(tpbBases) == 1:
                return tpbBases[0]
            else:
                return DirectSumBasis(tpbBases)
        else:
            return BuiltinBasis(name, dim, sparse)
        
    @classmethod
    def cast_from_basis(cls, basis, dim=None, sparse=None):
        #then just check to make sure consistent with `dim` & `sparse`
        if dim is not None:
            if isinstance(dim, _StateSpace):
                state_space = dim
                if hasattr(basis, 'state_space'):  # TODO - should *all* basis objects have a state_space?
                    assert(state_space.is_compatible_with(basis.state_space)), \
                        "Basis object has incompatible state space: %s != %s" % (str(state_space),
                                                                                    str(basis.state_space))
            else:  # assume dim is an integer
                assert(dim == basis.dim or dim == basis.elsize), \
                    "Basis object has unexpected dimension: %d != %d or %d" % (dim, basis.dim, basis.elsize)
        if sparse is not None:
            basis = basis.with_sparsity(sparse)
        return basis

    @classmethod
    def cast_from_arrays(cls, arrays, dim=None, sparse=None):
        b = ExplicitBasis(arrays, sparse=sparse)
        if dim is not None:
            assert(dim == b.dim), "Created explicit basis has unexpected dimension: %d vs %d" % (dim, b.dim)
        if sparse is not None:
            assert(sparse == b.sparse), "Basis object has unexpected sparsity: %s" % (b.sparse)
        return b

    @classmethod
    def cast(cls, arg, dim=None, sparse=None, classical_name='cl'):
        #print("DB: CAST = ",arg,dim)
        if isinstance(arg, Basis):
            return cls.cast_from_basis(arg, dim, sparse)
        if isinstance(arg, str):
            if isinstance(dim, _StateSpace):
                return cls.cast_from_name_and_statespace(arg, dim, sparse, classical_name)
            return cls.cast_from_name_and_dims(arg, dim, sparse)
        if (arg is None) or (hasattr(arg,'__len__') and len(arg) == 0):
            return ExplicitBasis([], [], "*Empty*", "Empty (0-element) basis", False, sparse)
            # ^ The original implementation would return this value under two conditions.
            #   Either arg was None, or isinstance(arg,(tuple,list,ndarray)) and len(arg) == 0.
            #   We're just slightly relaxing the type requirement by using this check instead.
        
        # At this point, original behavior would check that arg is a tuple, list, or ndarray.
        # Instead, we'll just require that arg[0] is well-defined. This is enough to discern
        # between the two cases we can still support.
        if isinstance(arg[0], _np.ndarray):
            return cls.cast_from_arrays(arg, dim, sparse)
        if len(arg[0]) == 2:
            compBases = [BuiltinBasis(subname, subdim, sparse) for (subname, subdim) in arg]
            return DirectSumBasis(compBases)

        raise ValueError("Can't cast %s to be a basis!" % str(type(arg)))


    def __init__(self, name, longname, real, sparse):
        super().__init__()
        self.name = name
        self.longname = longname
        self.real = real  # whether coefficients must be real (*not* whether elements are real - they're always complex)
        self.sparse = sparse  # whether elements are stored as sparse vectors/matrices

    @property
    def dim(self):
        """
        The dimension of the vector space this basis fully or partially
        spans.  Equivalently, the length of the `vector_elements` of the
        basis.
        """
        # dimension of vector space this basis fully or partially spans
        raise NotImplementedError("Derived classes must implement this!")

    @property
    def size(self):
        """
        The number of elements (or vector-elements) in the basis.
        """
        # number of elements (== dim if a *full* basis)
        raise NotImplementedError("Derived classes must implement this!")

    @property
    def elshape(self):
        """
        The shape of each element.  Typically either a length-1 or length-2
        tuple, corresponding to vector or matrix elements, respectively.
        Note that *vector elements* always have shape `(dim,)` (or `(dim,1)`
        in the sparse case).
        """
        # shape of "natural" elements - size may be > self.dim (to display naturally)
        raise NotImplementedError("Derived classes must implement this!")

    @property
    def elndim(self):
        """
        The number of element dimensions, i.e. `len(self.elshape)`

        Returns
        -------
        int
        """
        if self.elshape is None: return 0
        return len(self.elshape)

    @property
    def elsize(self):
        """
        The total element size, i.e. `product(self.elshape)`

        Returns
        -------
        int
        """
        if self.elshape is None: return 0
        return int(_np.prod(self.elshape))

    @property
    def first_element_is_identity(self):
        """
        True if the first element of this basis is *proportional* to the identity matrix, False otherwise.
        """
        if self.elndim != 2 or self.elshape[0] != self.elshape[1]: return False
        d = self.elshape[0]
        return _np.allclose(self.elements[0], _np.identity(d) * (_np.linalg.norm(self.elements[0]) / _np.sqrt(d)))

    def is_simple(self):
        """
        Whether the flattened-element vector space is the *same* space as the space this basis's vectors belong to.

        Returns
        -------
        bool
        """
        return self.elsize == self.dim

    def is_complete(self):
        """
        Whether this is a complete basis, i.e. this basis's vectors span the entire space that they live in.

        Returns
        -------
        bool
        """
        return self.dim == self.size

    def is_partial(self):
        """
        The negative of :meth:`is_complete`, effectively "is_incomplete".

        Returns
        -------
        bool
        """
        return not self.is_complete()

    @property
    def vector_elements(self):
        """
        The "vectors" of this basis, always 1D (sparse or dense) arrays.

        Returns
        -------
        list
            A list of 1D arrays.
        """
        if self.sparse:
            return [_sps.lil_matrix(el).reshape((self.elsize, 1)) for el in self.elements]
        else:
            # Use flatten (rather than ravel) to ensure a copy is made.
            return [el.flatten() for el in self.elements]

    def copy(self):
        """
        Make a copy of this Basis object.

        Returns
        -------
        Basis
        """
        return _copy.deepcopy(self)

    def with_sparsity(self, desired_sparsity):
        """
        Returns either this basis or a copy of it with the desired sparsity.

        If this basis has the desired sparsity it is simply returned.  If
        not, this basis is copied to one that does.

        Parameters
        ----------
        desired_sparsity : bool
            The sparsity (`True` for sparse elements, `False` for dense elements)
            that is desired.

        Returns
        -------
        Basis
        """
        if self.sparse == desired_sparsity:
            return self
        else:
            return self._copy_with_toggled_sparsity()

    def _copy_with_toggled_sparsity(self):
        raise NotImplementedError("Derived classes should implement this!")

    def __str__(self):
        return '{} (dim={}), {} elements of shape {} :\n{}'.format(
            self.longname, self.dim, self.size, self.elshape, ', '.join(self.labels))

    def __getitem__(self, index):
        if isinstance(index, str) and self.ellookup is not None:
            return self.ellookup[index]
        return self.elements[index]

    def __len__(self):
        return self.size

    def __eq__(self, other):
        return self.is_equivalent(other, sparseness_must_match=True)

    def is_equivalent(self, other, sparseness_must_match=True):
        """
        Tests whether this basis is equal to another basis, optionally ignoring sparseness.

        Parameters
        -----------
        other : Basis or str
            The basis to compare with.

        sparseness_must_match : bool, optional
            If `False` then comparison ignores differing sparseness, and this function
            returns `True` when the two bases are equal except for their `.sparse` values.

        Returns
        -------
        bool
        """
        otherIsBasis = isinstance(other, Basis)

        if otherIsBasis and sparseness_must_match and (self.sparse != other.sparse):
            return False  # sparseness mismatch => not equal when sparseness_must_match == True

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

    def create_transform_matrix(self, to_basis):
        """
        Get the matrix that transforms a vector from this basis to `to_basis`.

        Parameters
        ----------
        to_basis : Basis or string
            The basis to transform to or a built-in basis name.  In the latter
            case, a basis to transform to is built with the same structure as
            this basis but with all components constructed from the given name.

        Returns
        -------
        numpy.ndarray (even if basis is sparse)
        """
        #Note: construct to_basis as sparse this basis is sparse and
        # if to_basis is not already a Basis object
        if not isinstance(to_basis, Basis):
            to_basis = self.create_equivalent(to_basis)

        #Note same logic as matrixtools.safe_dot(...)
        if to_basis.sparse:
            return to_basis.from_std_transform_matrix.dot(self.to_std_transform_matrix)
        elif self.sparse:
            #return _sps.csr_matrix(to_basis.from_std_transform_matrix).dot(self.to_std_transform_matrix)
            return _np.dot(to_basis.from_std_transform_matrix, self.to_std_transform_matrix.toarray())
        else:
            return _np.dot(to_basis.from_std_transform_matrix, self.to_std_transform_matrix)

    def reverse_transform_matrix(self, from_basis):
        """
        Get the matrix that transforms a vector from `from_basis` to this basis.

        The reverse of :meth:`create_transform_matrix`.

        Parameters
        ----------
        from_basis : Basis or string
            The basis to transform from or a built-in basis name.  In the latter
            case, a basis to transform from is built with the same structure as
            this basis but with all components constructed from the given name.

        Returns
        -------
        numpy.ndarray (even if basis is sparse)
        """
        if not isinstance(from_basis, Basis):
            from_basis = self.create_equivalent(from_basis)

        #Note same logic as matrixtools.safe_dot(...)
        if self.sparse:
            return self.from_std_transform_matrix.dot(from_basis.to_std_transform_matrix)
        elif from_basis.sparse:
            #return _sps.csr_matrix(to_basis.from_std_transform_matrix).dot(self.to_std_transform_matrix)
            return _np.dot(self.from_std_transform_matrix, from_basis.to_std_transform_matrix.toarray())
        else:
            return _np.dot(self.from_std_transform_matrix, from_basis.to_std_transform_matrix)

    @lru_cache(maxsize=128)
    def is_normalized(self):
        """
        Check if a basis is normalized, meaning that Tr(Bi Bi) = 1.0.

        Available only to bases whose elements are *matrices* for now.

        Returns
        -------
        bool
        """
        if self.elndim == 2:
            for i, mx in enumerate(self.elements):
                t = _np.linalg.norm(mx) # == sqrt(tr(mx mx))
                if not _np.isclose(t, 1.0): return False
            return True
        elif self.elndim == 1:
            raise NotImplementedError("TODO: add code so this works for *vector*-valued bases too!")
        else:
            raise ValueError("I don't know what normalized means for elndim == %d!" % self.elndim)

    @property
    @lru_cache(maxsize=128)
    def to_std_transform_matrix(self):
        """
        Retrieve the matrix that transforms a vector from this basis to the standard basis of this basis's dimension.

        Returns
        -------
        numpy array or scipy.sparse.lil_matrix
            An array of shape `(dim, size)` where `dim` is the dimension
            of this basis (the length of its vectors) and `size` is the
            size of this basis (its number of vectors).
        """
        if self.sparse:
            toStd = _sps.lil_matrix((self.dim, self.size), dtype='complex')
        else:
            toStd = _np.zeros((self.dim, self.size), 'complex')

        for i, vel in enumerate(self.vector_elements):
            toStd[:, i] = vel
        return toStd

    @property
    @lru_cache(maxsize=128)
    def from_std_transform_matrix(self):
        """
        Retrieve the matrix that transforms vectors from the standard basis to this basis.

        Returns
        -------
        numpy array or scipy sparse matrix
            An array of shape `(size, dim)` where `dim` is the dimension
            of this basis (the length of its vectors) and `size` is the
            size of this basis (its number of vectors).
        """
        if self.sparse:
            if self.is_complete():
                return _spsl.inv(self.to_std_transform_matrix.tocsc()).tocsr()
            else:
                assert(self.size < self.dim), "Basis seems to be overcomplete: size > dimension!"
                # we'd need to construct a different pseudo-inverse if the above assert fails

                A = self.to_std_transform_matrix  # shape (dim,size) - should have indep *cols*
                Adag = A.getH()        # shape (size, dim)
                invAdagA = _spsl.inv(Adag.tocsr().dot(A.tocsc())).tocsr()
                return invAdagA.dot(Adag.tocsc())
        else:
            if self.is_complete():
                return _inv(self.to_std_transform_matrix)
            else:
                assert(self.size < self.dim), "Basis seems to be overcomplete: size > dimension!"
                # we'd need to construct a different pseudo-inverse if the above assert fails

                A = self.to_std_transform_matrix  # shape (dim,size) - should have indep *cols*
                Adag = A.transpose().conjugate()  # shape (size, dim)
                return _np.dot(_inv(_np.dot(Adag, A)), Adag)

    @property
    @lru_cache(maxsize=128)
    def to_elementstd_transform_matrix(self):
        """
        Get transformation matrix from this basis to the "element space".

        Get the matrix that transforms vectors in this basis (with length equal
        to the `dim` of this basis) to vectors in the "element space" - that
        is, vectors in the same standard basis that the *elements* of this basis
        are expressed in.

        Returns
        -------
        numpy array
            An array of shape `(element_dim, size)` where `element_dim` is the
            dimension, i.e. size, of the elements of this basis (e.g. 16 if the
            elements are 4x4 matrices) and `size` is the size of this basis (its
            number of vectors).
        """
        # This default implementation assumes that the (flattened) element space
        # *is* a standard representation of the vector space this basis or partial-basis
        # acts upon (this is *not* true for direct-sum bases, where the flattened
        # elements represent vectors in a larger "embedding" space (w/larger dim than actual space).
        assert(self.is_simple()), "Incorrectly using a simple-assuming implementation of to_elementstd_transform_matrix"
        return self.to_std_transform_matrix

    @property
    @lru_cache(maxsize=128)
    def from_elementstd_transform_matrix(self):  # OLD: get_expand_mx(self):
        """
        Get transformation matrix from "element space" to this basis.

        Get the matrix that transforms vectors in the "element space" - that
        is, vectors in the same standard basis that the *elements* of this basis
        are expressed in - to vectors in this basis (with length equal to the
        `dim` of this basis).

        Returns
        -------
        numpy array
            An array of shape `(size, element_dim)` where `element_dim` is the
            dimension, i.e. size, of the elements of this basis (e.g. 16 if the
            elements are 4x4 matrices) and `size` is the size of this basis (its
            number of vectors).
        """
        if self.sparse:
            raise NotImplementedError("from_elementstd_transform_matrix not implemented for sparse mode")  # (need pinv)
        return _np.linalg.pinv(self.to_elementstd_transform_matrix)

    def create_equivalent(self, builtin_basis_name):
        """
        Create an equivalent basis with components of type `builtin_basis_name`.

        Create a :class:`Basis` that is equivalent in structure & dimension to this
        basis but whose simple components (perhaps just this basis itself) is
        of the builtin basis type given by `builtin_basis_name`.

        Parameters
        ----------
        builtin_basis_name : str
            The name of a builtin basis, e.g. `"pp"`, `"gm"`, or `"std"`. Used to
            construct the simple components of the returned basis.

        Returns
        -------
        Basis
        """
        #This default implementation assumes that this basis is simple.
        assert(self.is_simple()), "Incorrectly using a simple-assuming implementation of create_equivalent()"
        return BuiltinBasis(builtin_basis_name, self.dim, sparse=self.sparse)

    #TODO: figure out if we actually need the return value from this function to
    # not have any components...  Maybe jamiolkowski.py needs this?  If it's
    # unnecessary, we can update these doc strings and perhaps TensorProdBasis's
    # implementation:
    def create_simple_equivalent(self, builtin_basis_name=None):
        """
        Create a basis of type `builtin_basis_name` whose elements are compatible with this basis.

        Create a simple basis *and* one without components (e.g. a
        :class:`TensorProdBasis`, is a simple basis w/components) of the
        builtin type specified whose dimension is compatible with the
        *elements* of this basis.  This function might also be named
        "element_equivalent", as it returns the `builtin_basis_name`-analogue
        of the standard basis that this basis's elements are expressed in.

        Parameters
        ----------
        builtin_basis_name : str, optional
            The name of the built-in basis to use.  If `None`, then a
            copy of this basis is returned (if it's simple) or this
            basis's name is used to try to construct a simple and
            component-free version of the same builtin-basis type.

        Returns
        -------
        Basis
        """
        #This default implementation assumes that this basis is simple.
        assert(self.is_simple()), "Incorrectly using a simple-assuming implementation of create_simple_equivalent()"
        if builtin_basis_name is None: return self.copy()
        else: return self.create_equivalent(builtin_basis_name)

    def is_compatible_with_state_space(self, state_space):
        """
        Checks whether this basis is compatible with a given state space.

        Parameters
        ----------
        state_space : StateSpace
            the state space to check.

        Returns
        -------
        bool
        """
        #FUTURE - need a way to deal with many qubits where total dim will overflow an int64
        #if self.state_space.dim is None:  # `None` indicates that dim is effectively infinite?
        return bool(self.dim == state_space.dim)


class LazyBasis(Basis):
    """
    A :class:`Basis` whose labels and elements that are constructed only when at least one of them is needed.

    This class is also used as a base class for higher-level, more specific basis classes.

    Parameters
    ----------
    name : string
        The name of the basis.  This can be anything, but is
        usually short and abbreviated.  There are several
        types of bases built into pyGSTi that can be constructed by
        this name.

    longname : string
        A more descriptive name for the basis.

    real : bool
        Elements and vector elements are always allowed to have complex
        entries.  This argument indicates whether the coefficients in the
        expression of an arbitrary vector in this basis must be real.  For
        example, if `real=True`, then when pyGSTi transforms a vector in
        some other basis to a vector in *this* basis, it will demand that
        the values of that vector (i.e. the coefficients which multiply
        this basis's elements to obtain a vector in the "standard" basis)
        are real.

    sparse : bool
        Whether the elements of `.elements` for this Basis are stored (when
        they are stored at all) as sparse matrices or vectors.


    Attributes
    ----------
    ellookup : dict
        A dictionary mapping basis element labels to the elements themselves, for fast element lookup.

    elements : numpy.ndarray
        The basis elements (sometimes different from the *vectors*)

    labels : list
        The basis labels
    """

    def __init__(self, name, longname, real, sparse):
        """
        Creates a new LazyBasis.  Parameters are the same as those to
        :meth:`Basis.__init__`.
        """
        self._elements = None        # "natural-shape" elements - can be vecs or matrices
        self._labels = None          # element labels
        self._ellookup = None        # fast element lookup
        super(LazyBasis, self).__init__(name, longname, real, sparse)

    def _lazy_build_elements(self):
        raise NotImplementedError("Derived classes must implement this function!")

    def _lazy_build_labels(self):
        raise NotImplementedError("Derived classes must implement this function!")

    @property
    def ellookup(self):
        """
        A dictionary mapping basis element labels to the elements themselves

        Returns
        -------
        dict
        """
        if self._ellookup is None:
            if self._elements is None:
                self._lazy_build_elements()
            if self._labels is None:
                self._lazy_build_labels()
            self._ellookup = {lbl: el for lbl, el in zip(self._labels, self._elements)}
        return self._ellookup

    @property
    def elements(self):
        """
        The basis elements (sometimes different from the *vectors*)

        Returns
        -------
        numpy.ndarray
        """
        if self._elements is None:
            self._lazy_build_elements()
        return self._elements

    @property
    def labels(self):
        """
        The basis labels

        Returns
        -------
        list
        """
        if self._labels is None:
            self._lazy_build_labels()
        return self._labels

    def __str__(self):
        if self._labels is None and self.dim > 64:
            return '{} (dim={}), {} elements of shape {} (not computed yet)'.format(
                self.longname, self.dim, self.size, self.elshape)
        else:
            return super(LazyBasis, self).__str__()


class ExplicitBasis(Basis):
    """
    A `Basis` whose elements are specified directly.

    All explicit bases are simple: their vector space is taken to be that
    of the the flattened elements unless separate `vector_elements` are given.

    Parameters
    ----------
    elements : numpy.ndarray
        The basis elements (sometimes different from the *vectors*)

    labels : list
        The basis labels

    name : str, optional
        The name of this basis.  If `None`, then a name will be
        automatically generated.

    longname : str, optional
        A more descriptive name for this basis.  If `None`, then the
        short `name` will be used.

    real : bool, optional
        Whether the coefficients in the expression of an arbitrary vector
        as a linear combination of this basis's elements must be real.

    sparse : bool, optional
        Whether the elements of this Basis are stored as sparse matrices or
        vectors.  If `None`, then this is automatically determined by the
        type of the initial object: `elements[0]` (`sparse=False` is used
        when `len(elements) == 0`).

    vector_elements : numpy.ndarray, optional
        A list or array of the 1D *vectors* corresponding to each element.
        If `None`, then the flattened elements are used as vectors.  The size
        of these vectors sets the dimension of the basis.

    Attributes
    ----------
    Count : int
        The number of custom bases, used for serialized naming
    """
    Count = 0  # The number of custom bases, used for serialized naming

    def __init__(self, elements, labels=None, name=None, longname=None, real=False, sparse=None, vector_elements=None):
        '''
        Create a new ExplicitBasis.

        Parameters
        ----------
        elements : iterable
            A list of the elements of this basis.

        labels : iterable, optional
            A list of the labels corresponding to the elements of `elements`.
            If given, `len(labels)` must equal `len(elements)`.

        name : str, optional
            The name of this basis.  If `None`, then a name will be
            automatically generated.

        longname : str, optional
            A more descriptive name for this basis.  If `None`, then the
            short `name` will be used.

        real : bool, optional
            Whether the coefficients in the expression of an arbitrary vector
            as a linear combination of this basis's elements must be real.

        sparse : bool, optional
            Whether the elements of this Basis are stored as sparse matrices or
            vectors.  If `None`, then this is automatically determined by the
            type of the initial object: `elements[0]` (`sparse=False` is used
            when `len(elements) == 0`).

        vector_elements : numpy.ndarray, optional
            A list or array of the 1D *vectors* corresponding to each element.
            If `None`, then the flattened elements are used as vectors.  The size
            of these vectors sets the dimension of the basis.
        '''
        if name is None:
            name = 'ExplicitBasis_{}'.format(ExplicitBasis.Count)
            if longname is None: longname = "Auto-named " + name
            ExplicitBasis.Count += 1
        elif longname is None: longname = name

        if labels is None: labels = ["E%d" % k for k in range(len(elements))]
        if (len(labels) != len(elements)):
            raise ValueError("Expected a list of %d labels but got: %s" % (len(elements), str(labels)))

        self.labels = tuple(labels)  # so hashable - see __hash__
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
                    if not _sps.issparse(el) or not _sps.isspmatrix_csr(el):  # needs to be CSR type for __hash__
                        el = _sps.csr_matrix(el)  # try to convert to a sparse matrix
                else:
                    if not isinstance(el, _np.ndarray):
                        el = _np.array(el)  # try to convert to a numpy array

                if elshape is None: elshape = el.shape
                else: assert(elshape == el.shape), "Inconsistent element shapes!"
                self.elements.append(el)
            dim = int(_np.prod(elshape))
        self.ellookup = {lbl: el for lbl, el in zip(self.labels, self.elements)}  # fast by-label element lookup

        if vector_elements is not None:
            assert(len(vector_elements) == size), "Must have the same number of `elements` and `vector_elements`"
            if sparse:
                self._vector_elements = [(el if _sps.issparse(el) else _sps.lil_matrix(el)) for el in vector_elements]
            else:
                self._vector_elements = _np.array(vector_elements)  # rows are *vectors*
            dim = self._vector_elements.shape[1]
        else:
            self._vector_elements = None

        self._dim = dim
        self._size = size
        self._elshape = elshape

        super(ExplicitBasis, self).__init__(name, longname, real, sparse)

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'name': self.name,
                      'longname': self.longname,
                      'real': self.real,
                      'sparse': self.sparse,
                      'labels': self.labels,
                      'elements': [self._encodemx(el) for el in self.elements],
                      'vector_elements': ([self._encodemx(vel) for vel in self._vector_elements]
                                          if (self._vector_elements is not None) else None)
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        vels = [cls._decodemx(vel) for vel in state['vector_elements']] \
            if (state.get('vector_elements', None) is not None) else None
        return cls([cls._decodemx(el) for el in state['elements']],
                   state['labels'], state['name'], state['longname'], state['real'],
                   state['sparse'], vels)

    @property
    def dim(self):
        """
        The dimension of the vector space this basis fully or partially
        spans.  Equivalently, the length of the `vector_elements` of the
        basis.
        """
        # dimension of vector space this basis fully or partially spans
        return self._dim

    @property
    def size(self):
        """
        The number of elements (or vector-elements) in the basis.
        """
        # number of elements (== dim if a *full* basis)
        return self._size

    @property
    def elshape(self):
        """
        The shape of each element.  Typically either a length-1 or length-2
        tuple, corresponding to vector or matrix elements, respectively.
        Note that *vector elements* always have shape `(dim,)` (or `(dim,1)`
        in the sparse case).
        """
        # shape of "natural" elements - size may be > self.dim (to display naturally)
        return self._elshape

    @property
    def vector_elements(self):
        """
        The "vectors" of this basis, always 1D (sparse or dense) arrays.

        Returns
        -------
        list
            A list of 1D arrays.
        """
        if self._vector_elements is not None:
            return self._vector_elements
        else:
            return Basis.vector_elements.fget(self)  # call base class get-property fn

    def _copy_with_toggled_sparsity(self):
        return ExplicitBasis(self.elements, self.labels, self.name, self.longname, self.real, not self.sparse,
                             self._vector_elements)

    def __hash__(self):
        if self.sparse:
            els_to_hash = tuple(((_np.round(el.data, 6).tobytes(), el.indices.tobytes(), el.indptr.tobytes())
                                 for el in self.elements))   # hash sparse matrices
        else:
            els_to_hash = tuple((_np.round(el, 6).tobytes() for el in self.elements))
        return hash((self.dim, self.elshape, self.sparse, self.labels, els_to_hash))  # TODO: hash vector els?
        # OLD return hash((self.name, self.dim, self.elshape, self.sparse))  # better?


class BuiltinBasis(LazyBasis):
    """
    A basis that is included within and integrated into pyGSTi.

    Such bases may, in most cases be represented merely by its name.  (In actuality,
    a dimension is also required, but this is often able to be inferred from context.)

    Parameters
    ----------
    name : {"pp", "gm", "std", "qt", "id", "cl", "sv"}
        Name of the basis to be created.

    dim_or_statespace : int or StateSpace
        The dimension of the basis to be created or the state space for which a
        basis should be created.  Note that when this is an integer it is the
        dimension of the *vectors*, which correspond to flattened elements
        in simple cases.  Thus, a 1-qubit basis would have dimension 2 in
        the state-vector (`name="sv"`) case and dimension 4 when
        constructing a density-matrix basis (e.g. `name="pp"`).

    sparse : bool, optional
        Whether basis elements should be stored as SciPy CSR sparse matrices
        or dense numpy arrays (the default).
    """

    def __init__(self, name, dim_or_statespace, sparse=False):
        from pygsti.baseobjs import statespace as _statespace
        assert(name in _basis_constructor_dict), "Unknown builtin basis name '%s'!" % name
        if sparse is None: sparse = False  # choose dense matrices by default (when sparsity is "unspecified")

        if name == 'cl':  # HACK for now, until we figure out better classical state spaces
            self.state_space = dim_or_statespace if isinstance(dim_or_statespace, _statespace.StateSpace) \
                else _statespace.ExplicitStateSpace([('L%d' % i,) for i in range(dim_or_statespace)])
        else:
            self.state_space = dim_or_statespace if isinstance(dim_or_statespace, _statespace.StateSpace) \
                else _statespace.default_space_for_dim(dim_or_statespace)

        longname = _basis_constructor_dict[name].longname
        real = _basis_constructor_dict[name].real
        self._first_element_is_identity = _basis_constructor_dict[name].first_element_is_identity

        super(BuiltinBasis, self).__init__(name, longname, real, sparse)

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'name': self.name,
                      'sparse': self.sparse,
                      'state_space': self.state_space.to_nice_serialization()
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        statespace = _StateSpace.from_nice_serialization(state['state_space'])
        return cls(state['name'], statespace, state['sparse'])

    @property
    def dim(self):
        """
        The dimension of the vector space this basis fully or partially
        spans.  Equivalently, the length of the `vector_elements` of the
        basis.
        """
        size, dim, elshape = _basis_constructor_dict[self.name].sizes(dim=self.state_space.dim, sparse=self.sparse)
        return dim

    @property
    def size(self):
        """
        The number of elements (or vector-elements) in the basis.
        """
        size, dim, elshape = _basis_constructor_dict[self.name].sizes(dim=self.state_space.dim, sparse=self.sparse)
        return size

    @property
    def elshape(self):
        """
        The shape of each element.  Typically either a length-1 or length-2
        tuple, corresponding to vector or matrix elements, respectively.
        Note that *vector elements* always have shape `(dim,)` (or `(dim,1)`
        in the sparse case).
        """
        size, dim, elshape = _basis_constructor_dict[self.name].sizes(dim=self.state_space.dim, sparse=self.sparse)

        #Check that sparse is True only when elements are *matrices*
        assert(not self.sparse or len(elshape) == 2), "`sparse == True` is only allowed for *matrix*-valued bases!"

        return elshape

    @property
    def first_element_is_identity(self):
        """
        True if the first element of this basis is *proportional* to the identity matrix, False otherwise.
        """
        return self._first_element_is_identity

    def __hash__(self):
        return hash((self.name, self.state_space, self.sparse))

    def _lazy_build_elements(self):
        f = _basis_constructor_dict[self.name].constructor
        cargs = {'dim': self.state_space.dim, 'sparse': self.sparse}
        self._elements = _np.array(f(**cargs))  # a list of (dense) mxs -> ndarray (possibly sparse in future?)
        assert(len(self._elements) == self.size), "Logic error: wrong number of elements were created!"

    def _lazy_build_labels(self):
        f = _basis_constructor_dict[self.name].labeler
        cargs = {'dim': self.state_space.dim, 'sparse': self.sparse}
        self._labels = f(**cargs)

    def _copy_with_toggled_sparsity(self):
        return BuiltinBasis(self.name, self.state_space, not self.sparse)

    def is_equivalent(self, other, sparseness_must_match=True):
        """
        Tests whether this basis is equal to another basis, optionally ignoring sparseness.

        Parameters
        -----------
        other : Basis or str
            The basis to compare with.

        sparseness_must_match : bool, optional
            If `False` then comparison ignores differing sparseness, and this function
            returns `True` when the two bases are equal except for their `.sparse` values.

        Returns
        -------
        bool
        """
        if isinstance(other, BuiltinBasis):  # then can compare quickly
            return ((self.name == other.name)
                    and (self.state_space == other.state_space)
                    and (not sparseness_must_match or (self.sparse == other.sparse)))
        elif isinstance(other, str):
            return self.name == other  # see if other is a string equal to our name
        else:
            return LazyBasis.is_equivalent(self, other)


class DirectSumBasis(LazyBasis):
    """
    A basis that is the direct sum of one or more "component" bases.

    Elements of this basis are the union of the basis elements on each component,
    each embedded into a common block-diagonal structure where each component
    occupies its own block.  Thus, when there is more than one component, a
    `DirectSumBasis` is not a simple basis because the size of its elements
    is larger than the size of its vector space (which corresponds to just
    the diagonal blocks of its elements).

    Parameters
    ----------
    component_bases : iterable
        A list of the component bases.  Each list elements may be either
        a Basis object or a tuple of arguments to :func:`Basis.cast`,
        e.g. `('pp',4)`.

    name : str, optional
        The name of this basis.  If `None`, the names of the component bases
        joined with "+" is used.

    longname : str, optional
        A longer description of this basis.  If `None`, then a long name is
        automatically generated.

    Attributes
    ----------
    vector_elements : list
        The "vectors" of this basis, always 1D (sparse or dense) arrays.
    """

    def __init__(self, component_bases, name=None, longname=None):
        '''
        Create a new DirectSumBasis - a basis for a space that is the direct-sum
        of the spaces spanned by other "component" bases.

        Parameters
        ----------
        component_bases : iterable
            A list of the component bases.  Each list elements may be either
            a Basis object or a tuple of arguments to :func:`Basis.cast`,
            e.g. `('pp',4)`.

        name : str, optional
            The name of this basis.  If `None`, the names of the component bases
            joined with "+" is used.

        longname : str, optional
            A longer description of this basis.  If `None`, then a long name is
            automatically generated.
        '''
        assert(len(component_bases) > 0), "Must supply at least one component basis"

        self.component_bases = []
        self._vector_elements = None  # vectorized elements: 1D arrays

        for compbasis in component_bases:
            if isinstance(compbasis, Basis):
                self.component_bases.append(compbasis)
            else:
                #compbasis can be a list/tuple of args to Basis.cast, e.g. ('pp',2)
                self.component_bases.append(Basis.cast(*compbasis))

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
        super(DirectSumBasis, self).__init__(name, longname, real, sparse)

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'name': self.name,
                      'longname': self.longname,
                      'component_bases': [b.to_nice_serialization() for b in self.component_bases]
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        component_bases = [Basis.from_nice_serialization(b) for b in state['component_bases']]
        return cls(component_bases, state['name'], state['longname'])

    @property
    def dim(self):
        """
        The dimension of the vector space this basis fully or partially
        spans.  Equivalently, the length of the `vector_elements` of the
        basis.
        """
        return sum([c.dim for c in self.component_bases])

    @property
    def size(self):
        """
        The number of elements (or vector-elements) in the basis.
        """
        return sum([c.size for c in self.component_bases])

    @property
    def elshape(self):
        """
        The shape of each element.  Typically either a length-1 or length-2
        tuple, corresponding to vector or matrix elements, respectively.
        Note that *vector elements* always have shape `(dim,)` (or `(dim,1)`
        in the sparse case).
        """
        elndim = len(self.component_bases[0].elshape)
        assert(all([len(c.elshape) == elndim for c in self.component_bases])
               ), "Inconsistent element ndims among component bases!"
        return tuple([sum([c.elshape[k] for c in self.component_bases]) for k in range(elndim)])

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
                    mx = _sps.lil_matrix((self.dim, 1), dtype='complex')
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

    def _copy_with_toggled_sparsity(self):
        return DirectSumBasis([cb._copy_with_toggled_sparsity() for cb in self.component_bases],
                              self.name, self.longname)

    def is_equivalent(self, other, sparseness_must_match=True):
        """
        Tests whether this basis is equal to another basis, optionally ignoring sparseness.

        Parameters
        -----------
        other : Basis or str
            The basis to compare with.

        sparseness_must_match : bool, optional
            If `False` then comparison ignores differing sparseness, and this function
            returns `True` when the two bases are equal except for their `.sparse` values.

        Returns
        -------
        bool
        """
        otherIsBasis = isinstance(other, DirectSumBasis)
        if not otherIsBasis: return False  # can't be equal to a non-DirectSumBasis
        if len(self.component_bases) != len(other.component_bases): return False
        return all([c1.is_equivalent(c2, sparseness_must_match)
                    for (c1, c2) in zip(self.component_bases, other.component_bases)])

    @property
    def vector_elements(self):
        """
        The "vectors" of this basis, always 1D (sparse or dense) arrays.

        Returns
        -------
        list
        """
        if self._vector_elements is None:
            self._lazy_build_vector_elements()
        return self._vector_elements

    @property
    @lru_cache(maxsize=128)
    def to_std_transform_matrix(self):
        """
        Retrieve the matrix that transforms a vector from this basis to the standard basis of this basis's dimension.

        Returns
        -------
        numpy array or scipy.sparse.lil_matrix
            An array of shape `(dim, size)` where `dim` is the dimension
            of this basis (the length of its vectors) and `size` is the
            size of this basis (its number of vectors).
        """
        if self.sparse:
            toStd = _sps.lil_matrix((self.dim, self.size), dtype='complex')
        else:
            toStd = _np.zeros((self.dim, self.size), 'complex')

        #use vector elements, which are not just flattened elements
        # (and are computed separately)
        for i, vel in enumerate(self.vector_elements):
            toStd[:, i] = vel
        return toStd

    @property
    @lru_cache(maxsize=128)
    def to_elementstd_transform_matrix(self):
        """
        Get transformation matrix from this basis to the "element space".

        Get the matrix that transforms vectors in this basis (with length equal
        to the `dim` of this basis) to vectors in the "element space" - that
        is, vectors in the same standard basis that the *elements* of this basis
        are expressed in.

        Returns
        -------
        numpy array
            An array of shape `(element_dim, size)` where `element_dim` is the
            dimension, i.e. size, of the elements of this basis (e.g. 16 if the
            elements are 4x4 matrices) and `size` is the size of this basis (its
            number of vectors).
        """
        assert(not self.sparse), "to_elementstd_transform_matrix not implemented for sparse mode"
        expanddim = self.elsize  # == _np.prod(self.elshape)
        if self.sparse:
            toSimpleStd = _sps.lil_matrix((expanddim, self.size), dtype='complex')
        else:
            toSimpleStd = _np.zeros((expanddim, self.size), 'complex')

        for i, el in enumerate(self.elements):
            if self.sparse:
                vel = _sps.lil_matrix(el.reshape(-1, 1))  # sparse vector == sparse n x 1 matrix
            else:
                vel = el.ravel()
            toSimpleStd[:, i] = vel
        return toSimpleStd

    def create_equivalent(self, builtin_basis_name):
        """
        Create an equivalent basis with components of type `builtin_basis_name`.

        Create a Basis that is equivalent in structure & dimension to this
        basis but whose simple components (perhaps just this basis itself) is
        of the builtin basis type given by `builtin_basis_name`.

        Parameters
        ----------
        builtin_basis_name : str
            The name of a builtin basis, e.g. `"pp"`, `"gm"`, or `"std"`. Used to
            construct the simple components of the returned basis.

        Returns
        -------
        DirectSumBasis
        """
        equiv_components = [c.create_equivalent(builtin_basis_name) for c in self.component_bases]
        return DirectSumBasis(equiv_components)

    def create_simple_equivalent(self, builtin_basis_name=None):
        """
        Create a basis of type `builtin_basis_name` whose elements are compatible with this basis.

        Create a simple basis *and* one without components (e.g. a
        :class:`TensorProdBasis`, is a simple basis w/components) of the
        builtin type specified whose dimension is compatible with the
        *elements* of this basis.  This function might also be named
        "element_equivalent", as it returns the `builtin_basis_name`-analogue
        of the standard basis that this basis's elements are expressed in.

        Parameters
        ----------
        builtin_basis_name : str, optional
            The name of the built-in basis to use.  If `None`, then a
            copy of this basis is returned (if it's simple) or this
            basis's name is used to try to construct a simple and
            component-free version of the same builtin-basis type.

        Returns
        -------
        Basis
        """
        if builtin_basis_name is None:
            builtin_basis_name = self.name  # default
            if len(self.component_bases) > 0:
                first_comp_name = self.component_bases[0].name
                if all([c.name == first_comp_name for c in self.component_bases]):
                    builtin_basis_name = first_comp_name  # if all components have the same name
        return BuiltinBasis(builtin_basis_name, self.elsize, sparse=self.sparse)  # Note: changes dimension


class TensorProdBasis(LazyBasis):
    """
    A Basis that is the tensor product of one or more "component" bases.

    The elements of a TensorProdBasis consist of all tensor products of
    component basis elements (respecting the order given).  The components
    of a TensorProdBasis must be simple bases so that kronecker products
    can be used to produce the parent basis's elements.

    A TensorProdBasis is a "simple" basis in that its flattened elements
    do correspond to its vectors.

    Parameters
    ----------
    component_bases : iterable
        A list of the component bases.  Each list elements may be either
        a Basis object or a tuple of arguments to :func:`Basis.cast`,
        e.g. `('pp',4)`.

    name : str, optional
        The name of this basis.  If `None`, the names of the component bases
        joined with "*" is used.

    longname : str, optional
        A longer description of this basis.  If `None`, then a long name is
        automatically generated.
    """

    def __init__(self, component_bases, name=None, longname=None):
        '''
        Create a new TensorProdBasis whose elements are the tensor products
        of the elements of a set of "component" bases.

        Parameters
        ----------
        component_bases : iterable
            A list of the component bases.  Each list elements may be either
            a Basis object or a tuple of arguments to :func:`Basis.cast`,
            e.g. `('pp',4)`.

        name : str, optional
            The name of this basis.  If `None`, the names of the component bases
            joined with "*" is used.

        longname : str, optional
            A longer description of this basis.  If `None`, then a long name is
            automatically generated.
        '''
        assert(len(component_bases) > 0), "Must supply at least one component basis"

        self.component_bases = []
        for compbasis in component_bases:
            if isinstance(compbasis, Basis):
                self.component_bases.append(compbasis)
            else:
                #compbasis can be a list/tuple of args to Basis.cast, e.g. ('pp',2)
                self.component_bases.append(Basis.cast(*compbasis))

        if name is None:
            name = "*".join([c.name for c in self.component_bases])
        if longname is None:
            longname = "Tensor-product basis with components " + ", ".join(
                [c.name for c in self.component_bases])

        real = all([c.real for c in self.component_bases])
        sparse = all([c.sparse for c in self.component_bases])
        #assert(all([c.real == real for c in self.component_bases])), "Inconsistent `real` value among component bases!"
        assert(all([c.sparse == sparse for c in self.component_bases])), "Inconsistent sparsity among component bases!"

        super(TensorProdBasis, self).__init__(name, longname, real, sparse)

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'name': self.name,
                      'longname': self.longname,
                      'component_bases': [b.to_nice_serialization() for b in self.component_bases]
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        component_bases = [Basis.from_nice_serialization(b) for b in state['component_bases']]
        return cls(component_bases, state['name'], state['longname'])

    @property
    def dim(self):
        """
        The dimension of the vector space this basis fully or partially
        spans.  Equivalently, the length of the `vector_elements` of the
        basis.
        """
        dim = int(_np.prod([c.dim for c in self.component_bases]))

        #NOTE: this is actually to restrictive -- what we need is a test/flag for whether the elements of a
        # basis are in their "natrual" representation where it makes sense to take tensor products.  For
        # example, a direct-sum basis may hold elements in a compact way that violate this... but I'm not sure if they
        # do and this needs to be checked.  For now, we could just disable this overly-restrictive assert:
        assert(all([c.is_simple() for c in self.component_bases])), \
            "Components of a tensor product basis must be *simple* (have vector-dimension == size of elements)"
        # because we use the natural representation to take tensor (kronecker) products.
        # Note: this assertion also means dim == product(component_elsizes) == elsize, so basis is *simple*

        return dim

    @property
    def size(self):
        """
        The number of elements (or vector-elements) in the basis.
        """
        return int(_np.prod([c.size for c in self.component_bases]))

    @property
    def elshape(self):
        """
        The shape of each element.  Typically either a length-1 or length-2
        tuple, corresponding to vector or matrix elements, respectively.
        Note that *vector elements* always have shape `(dim,)` (or `(dim,1)`
        in the sparse case).
        """
        elndim = max([c.elndim for c in self.component_bases])
        elshape = [1] * elndim
        for c in self.component_bases:
            off = elndim - c.elndim
            for k, d in enumerate(c.elshape):
                elshape[k + off] *= d
        return tuple(elshape)

    def __hash__(self):
        return hash(tuple((hash(comp) for comp in self.component_bases)))

    def _lazy_build_elements(self):
        #LAZY building of elements (in case we never need them)
        if self.sparse:
            compMxs = [None] * self.size
        else:
            compMxs = _np.zeros((self.size,) + self.elshape, 'complex')

        #Take kronecker product of *natural* reps of component-basis elements
        # then reshape to vectors at the end.  This requires that the vector-
        # dimension of the component spaces equals the "natural space" dimension.
        comp_els = [c.elements for c in self.component_bases]
        for i, factors in enumerate(_itertools.product(*comp_els)):
            if self.sparse:
                M = _sps.identity(1, 'complex', 'csr')
                for f in factors:
                    M = _sps.kron(M, f, 'csr')
            else:
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

    def _copy_with_toggled_sparsity(self):
        return TensorProdBasis([cb._copy_with_toggled_sparsity() for cb in self.component_bases],
                               self.name, self.longname)

    def is_equivalent(self, other, sparseness_must_match=True):
        """
        Tests whether this basis is equal to another basis, optionally ignoring sparseness.

        Parameters
        -----------
        other : Basis or str
            The basis to compare with.

        sparseness_must_match : bool, optional
            If `False` then comparison ignores differing sparseness, and this function
            returns `True` when the two bases are equal except for their `.sparse` values.

        Returns
        -------
        bool
        """
        otherIsBasis = isinstance(other, TensorProdBasis)
        if not otherIsBasis: return False  # can't be equal to a non-DirectSumBasis
        if len(self.component_bases) != len(other.component_bases): return False
        if self.sparse != other.sparse: return False
        return all([c1.is_equivalent(c2, sparseness_must_match)
                    for (c1, c2) in zip(self.component_bases, other.component_bases)])

    def create_equivalent(self, builtin_basis_name):
        """
        Create an equivalent basis with components of type `builtin_basis_name`.

        Create a Basis that is equivalent in structure & dimension to this
        basis but whose simple components (perhaps just this basis itself) is
        of the builtin basis type given by `builtin_basis_name`.

        Parameters
        ----------
        builtin_basis_name : str
            The name of a builtin basis, e.g. `"pp"`, `"gm"`, or `"std"`. Used to
            construct the simple components of the returned basis.

        Returns
        -------
        TensorProdBasis
        """
        # FUTURE: we may want a way of creating a 'std' equivalent of tensor product bases that include classical lines.
        # This is a part of what woudl go into that... but it's not complete.
        # if builtin_basis_name == 'std':  # special case when we change classical components to 'cl'
        #     equiv_components = []
        #     for c in self.component_bases:
        #         if c.elndim == 1: equiv_components.append(c.create_equivalent('cl'))
        #         else: equiv_components.append(c.create_equivalent('std'))
        # else:
        equiv_components = [c.create_equivalent(builtin_basis_name) for c in self.component_bases]
        return TensorProdBasis(equiv_components)

    def create_simple_equivalent(self, builtin_basis_name=None):
        """
        Create a basis of type `builtin_basis_name` whose elements are compatible with this basis.

        Create a simple basis *and* one without components (e.g. a
        :class:`TensorProdBasis`, is a simple basis w/components) of the
        builtin type specified whose dimension is compatible with the
        *elements* of this basis.  This function might also be named
        "element_equivalent", as it returns the `builtin_basis_name`-analogue
        of the standard basis that this basis's elements are expressed in.

        Parameters
        ----------
        builtin_basis_name : str, optional
            The name of the built-in basis to use.  If `None`, then a
            copy of this basis is returned (if it's simple) or this
            basis's name is used to try to construct a simple and
            component-free version of the same builtin-basis type.

        Returns
        -------
        Basis
        """
        #if builtin_basis_name == 'std':  # special case when we change classical components to 'clmx'
        #    equiv_components = []
        #    for c in self.component_bases:
        #        if c.elndim == 1: equiv_components.append(BuiltinBasis('clmx', c.dim**2, sparse=self.sparse))
        #        # c.create_simple_equivalent('clmx'))
        #        else: equiv_components.append(c.create_simple_equivalent('std'))
        #    expanded_basis = TensorProdBasis(equiv_components)
        #    return BuiltinBasis('std', expanded_basis.elsize, sparse=expanded_basis.sparse)

        if builtin_basis_name is None:
            builtin_basis_name = self.name  # default
            if len(self.component_bases) > 0:
                first_comp_name = self.component_bases[0].name
                if all([c.name == first_comp_name for c in self.component_bases]):
                    builtin_basis_name = first_comp_name  # if all components have the same name
        return BuiltinBasis(builtin_basis_name, self.elsize, sparse=self.sparse)


class EmbeddedBasis(LazyBasis):
    """
    A basis that embeds a basis for a smaller state space within a larger state space.

    The elements of an EmbeddedBasis are therefore just embedded versions
    of the elements of the basis that is embedded.

    Parameters
    ----------
    basis_to_embed : Basis
        The basis being embedded.

    state_space_labels : StateSpaceLabels
        An object describing the struture of the entire state space.

    target_labels : list or tuple
        The labels contained in `stateSpaceLabels` which demarcate the
        portions of the state space acted on by `basis_to_embed`.

    name : str, optional
        The name of this basis.  If `None`, the names of `basis_to_embed`
        is joined with ':' characters to the elements of `target_labels`.

    longname : str, optional
        A longer description of this basis.  If `None`, then a long name is
        automatically generated.
    """

    @classmethod
    def embed_label(cls, lbl, target_labels):
        """
        Gets the EmbeddedBasis label for `lbl`.

        Convenience method that gives the EmbeddedBasis label for `lbl`
        without needing to construct the `EmbeddedBasis`.  E.g. `"XX:1,2"`.

        Parameters
        ----------
        lbl : str
            Un-embedded basis element label, e.g. `"XX"`.

        target_labels : tuple
            The target state space labels upon which this basis element
            will be embedded, e.g. `(1,2)`

        Returns
        -------
        str
            The embedded-basis-element label as an EmbeddedBasis would
            assign it.  E.g. `"XX:1,2"`.
        """
        return "%s:%s" % (lbl, ",".join(map(str, target_labels)))

    @classmethod
    def unembed_label(cls, lbl, target_labels):
        """
        Convenience method that performs the reverse of :meth:`embed_label`

        Parameters
        ----------
        lbl : str
            Embedded basis element label, e.g. `"XX:1,2"`.

        target_labels : tuple
            The target state space labels upon which this basis element
            will be embedded, e.g. `(1,2)`

        Returns
        -------
        str
            The un-embedded label, e.g. `"XX"`.
        """
        suffix = ":" + ",".join(map(str, target_labels))
        if lbl.endswith(suffix):
            return lbl[:-len(suffix)]
        else:
            raise ValueError("Cannot unembed '%s' - doesn't end in '%s'!" % (lbl, suffix))

    def __init__(self, basis_to_embed, state_space, target_labels, name=None, longname=None):
        '''
        Create a new EmbeddedBasis.

        Parameters
        ----------
        basis_to_embed : Basis
            The basis being embedded.

        state_space : StateSpace
            An object describing the struture of the entire state space.

        target_labels : list or tuple
            The labels contained in `stateSpaceLabels` which demarcate the
            portions of the state space acted on by `basis_to_embed`.

        name : str, optional
            The name of this basis.  If `None`, the names of `basis_to_embed`
            is joined with ':' characters to the elements of `target_labels`.

        longname : str, optional
            A longer description of this basis.  If `None`, then a long name is
            automatically generated.
        '''
        from pygsti.baseobjs.statespace import StateSpace as _StateSpace
        self.embedded_basis = basis_to_embed
        self.target_labels = target_labels
        self.state_space = _StateSpace.cast(state_space)

        if name is None:
            name = ':'.join((basis_to_embed.name,) + tuple(map(str, target_labels)))
        if longname is None:
            longname = "Embedded %s basis as %s within %s" % \
                (basis_to_embed.name, ':'.join(map(str, target_labels)), str(self.state_space))

        real = basis_to_embed.real
        sparse = basis_to_embed.sparse

        super(EmbeddedBasis, self).__init__(name, longname, real, sparse)

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'name': self.name,
                      'longname': self.longname,
                      'state_space': self.state_space.to_nice_serialization(),
                      'embedded_basis': self.embedded_basis.to_nice_serialization()
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        basis_to_embed = Basis.from_nice_serialization(state['embedded_basis'])
        state_space = _StateSpace.from_nice_serialization(state['state_space'])
        return cls(basis_to_embed, state_space, state['target_labels'], state['name'], state['longname'])

    @property
    def dim(self):
        """
        The dimension of the vector space this basis fully or partially
        spans.  Equivalently, the length of the `vector_elements` of the
        basis.
        """
        return self.state_space.dim

    @property
    def size(self):
        """
        The number of elements (or vector-elements) in the basis.
        """
        return self.embedded_basis.size

    @property
    def elshape(self):
        """
        The shape of each element.  Typically either a length-1 or length-2
        tuple, corresponding to vector or matrix elements, respectively.
        Note that *vector elements* always have shape `(dim,)` (or `(dim,1)`
        in the sparse case).
        """
        elndim = self.embedded_basis.elndim
        if elndim == 2:  # a "matrix" basis
            d = int(_np.sqrt(self.dim))
            assert(d**2 == self.dim), \
                "Dimension of state_space must be a perfect square when embedding a matrix basis"
            elshape = (d, d)
        elif elndim == 1:
            elshape = (self.dim,)
        else:
            raise ValueError("Can only embed bases with .elndim == 1 or 2 (received %d)!" % elndim)
        return elshape

    def __hash__(self):
        return hash(tuple(hash(self.embedded_basis), self.target_labels, self.state_space))

    def _lazy_build_elements(self):
        """ Take a dense or sparse basis matrix and embed it. """
        #LAZY building of elements (in case we never need them)
        if self.elndim == 2:  # then use EmbeddedOp to do matrix
            from ..modelmembers.operations import StaticArbitraryOp
            from ..modelmembers.operations import EmbeddedOp
            sslbls = self.state_space.copy()
            sslbls.reduce_dims_densitymx_to_state_inplace()  # because we're working with basis matrices not gates

            if self.sparse:
                self._elements = []
                for spmx in self.embedded_basis.elements:
                    mxAsOp = StaticArbitraryOp(spmx.to_dense(), evotype='statevec')
                    self._elements.append(EmbeddedOp(sslbls, self.target_labels,
                                                     mxAsOp).to_sparse())
            else:
                self._elements = _np.zeros((self.size,) + self.elshape, 'complex')
                for i, mx in enumerate(self.embedded_basis.elements):
                    self._elements[i] = EmbeddedOp(
                        sslbls, self.target_labels, StaticArbitraryOp(mx, evotype='statevec')
                    ).to_dense(on_space='HilbertSchmidt')
        else:
            # we need to perform embedding using vectors rather than matrices - doable, but
            # not needed yet, so defer implementation to later.
            raise NotImplementedError("Embedding *vector*-type bases not implemented yet")

    def _lazy_build_labels(self):
        self._labels = [EmbeddedBasis.embed_label(lbl, self.target_labels)
                        for lbl in self.embedded_basis.labels]

    def _copy_with_toggled_sparsity(self):
        return EmbeddedBasis(self.embedded_basis._copy_with_toggled_sparsity(),
                             self.state_space,
                             self.target_labels,
                             self.name, self.longname)

    def is_equivalent(self, other, sparseness_must_match=True):
        """
        Tests whether this basis is equal to another basis, optionally ignoring sparseness.

        Parameters
        -----------
        other : Basis or str
            The basis to compare with.

        sparseness_must_match : bool, optional
            If `False` then comparison ignores differing sparseness, and this function
            returns `True` when the two bases are equal except for their `.sparse` values.

        Returns
        -------
        bool
        """
        otherIsBasis = isinstance(other, EmbeddedBasis)
        if not otherIsBasis: return False  # can't be equal to a non-EmbeddedBasis
        if self.target_labels != other.target_labels or self.state_space != other.state_space:
            return False
        return self.embedded_basis.is_equivalent(other.embedded_basis, sparseness_must_match)

    def create_equivalent(self, builtin_basis_name):
        """
        Create an equivalent basis with components of type `builtin_basis_name`.

        Create a Basis that is equivalent in structure & dimension to this
        basis but whose simple components (perhaps just this basis itself) is
        of the builtin basis type given by `builtin_basis_name`.

        Parameters
        ----------
        builtin_basis_name : str
            The name of a builtin basis, e.g. `"pp"`, `"gm"`, or `"std"`. Used to
            construct the simple components of the returned basis.

        Returns
        -------
        EmbeddedBasis
        """
        equiv_embedded = self.embedded_basis.create_equivalent(builtin_basis_name)
        return EmbeddedBasis(equiv_embedded, self.state_space, self.target_labels)

    def create_simple_equivalent(self, builtin_basis_name=None):
        """
        Create a basis of type `builtin_basis_name` whose elements are compatible with this basis.

        Create a simple basis *and* one without components (e.g. a
        :class:`TensorProdBasis`, is a simple basis w/components) of the
        builtin type specified whose dimension is compatible with the
        *elements* of this basis.  This function might also be named
        "element_equivalent", as it returns the `builtin_basis_name`-analogue
        of the standard basis that this basis's elements are expressed in.

        Parameters
        ----------
        builtin_basis_name : str, optional
            The name of the built-in basis to use.  If `None`, then a
            copy of this basis is returned (if it's simple) or this
            basis's name is used to try to construct a simple and
            component-free version of the same builtin-basis type.

        Returns
        -------
        Basis
        """
        if builtin_basis_name is None:
            builtin_basis_name = self.embedded_basis.name  # default
        return BuiltinBasis(builtin_basis_name, self.elsize, sparse=self.sparse)
