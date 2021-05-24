"""
Defines OrderedDict-derived classes used to store specific pyGSTi objects
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
import copy as _copy
import numbers as _numbers
import sys as _sys


class StateSpace(object):
    """
    Base class for defining a state space (Hilbert or Hilbert-Schmidt space).
    """

    @property
    def udim(self):
        pass
    
    @property
    def dim(self):
        pass

    @property
    def num_qubits(self):
        pass

    @property
    def num_tensor_prod_blocks(self):
        """
        Get the number of tensor-product blocks which are direct-summed to get the final state space.

        Returns
        -------
        int
        """
        return 0

    def tensor_product_block_labels(self, i_tpb):
        """
        Get the labels for the `iTBP`-th tensor-product block.

        Parameters
        ----------
        i_tpb : int
            Tensor-product block index.

        Returns
        -------
        tuple
        """
        raise ValueError("Invalid tensor product block index: %d" % i_tpb)

    @property
    def tensor_product_blocks_labels(self):
        """
        Get the labels for all the tensor-product blocks.

        Returns
        -------
        tuple
        """
        return ()

    def tensor_product_block_dimensions(self, i_tpb):
        #return list of dimensions corresponding to labels
        pass
        #return self.tpb_dims  # TODO
        #_np.product(state_space.tensor_product_block_dims(k))
    
    @property
    def tensor_product_blocks_dimensions(self):
        #return list of lists of dimensions corresponding to labels for each TPB
        pass
        #return self.tpb_dims  # TODO
        #_np.product(state_space.tensor_product_block_dims(k))


    def copy(self):
        pass

    def is_compatible_with(self, other_state_space):
        raise NotImplementedError("TODO!!")

    def label_dimension(self, label):
        pass  # like .labeldims

    def label_tensor_product_block_index(self, label):
        pass # like .tpb_index


class QubitSpace(StateSpace):
    """
    A state space consisting of N qubits.
    """

    def __init__(self, nqubits_or_labels):
        pass
    
    def qubit_labels(self):
        pass

    def classical_labels(self):
        pass


class CustomStateSpace(StateSpace):
    """
    A customizable definition of a state space.

    A CustomStateSpace object describes, using string/int labels, how an entire
    Hilbert state space is decomposed into the direct sum of terms which
    themselves are tensor products of smaller (typically qubit-sized) Hilbert
    spaces.

    Parameters
    ----------
    label_list : str or int or iterable
        Most generally, this can be a list of tuples, where each tuple
        contains the state-space labels (which can be strings or integers)
        for a single "tensor product block" formed by taking the tensor
        product of the spaces asociated with the labels.  The full state
        space is the direct sum of all the tensor product blocks.
        E.g. `[('Q0','Q1'), ('Q2',)]`.

        If just an iterable of labels is given, e.g. `('Q0','Q1')`, it is
        assumed to specify the first and only tensor product block.

        If a single state space label is given, e.g. `'Q2'`, then it is
        assumed to completely specify the first and only tensor product
        block.

    dims : int or iterable, optional
        The dimension of each state space label as an integer, tuple of
        integers, or list or tuples of integers to match the structure
        of `label_list` (i.e., if `label_list=('Q0','Q1')` then `dims` should
        be a tuple of 2 integers).  Values specify state-space dimensions: 2
        for a qubit, 3 for a qutrit, etc.  If None, then the dimensions are
        inferred, if possible, from the following naming rules:

        - if the label starts with 'L', dim=1 (a single Level)
        - if the label starts with 'Q' OR is an int, dim=2 (a Qubit)
        - if the label starts with 'T', dim=3 (a quTrit)

    types : str or iterable, optional
        A list of label types, either `'Q'` or `'C'` for "quantum" and
        "classical" respectively, indicating the type of state-space
        associated with each label.  Like `dims`, `types` must match
        the structure of `label_list`.  A quantum state space of dimension
        `d` is a `d`-by-`d` density matrix, whereas a classical state space
        of dimension d is a vector of `d` probabilities.  If `None`, then
        all labels are assumed to be quantum.

    evotype : {"densitymx","statevec","stabilizer","svterm","cterm"}
        The evolution type that this state-space will be used with.  This
        information is needed just to select the appropriate default
        dimensions, e.g. whether a qubit has a 2- or 4-dimensional state
        space.
    """

    @classmethod
    def cast(cls, obj):
        """
        Casts `obj` into a :class:`CustomStateSpace` object if possible.

        If `obj` is already of this type, it is simply returned without modification.

        Parameters
        ----------
        obj : CustomStateSpace or list
            Either an already-built state space labels object or a list of labels
            as would be provided to the first argument of :method:`CustomStateSpace.__init__`.

        Returns
        -------
        CustomStateSpace
        """
        if isinstance(obj, cls):
            return obj
        else:
            return cls(obj)

    def __init__(self, label_list, dims=None, types=None, evotype="densitymx"):
        """
        Creates a new CustomStateSpace object.

        Parameters
        ----------
        label_list : str or int or iterable
            Most generally, this can be a list of tuples, where each tuple
            contains the state-space labels (which can be strings or integers)
            for a single "tensor product block" formed by taking the tensor
            product of the spaces asociated with the labels.  The full state
            space is the direct sum of all the tensor product blocks.
            E.g. `[('Q0','Q1'), ('Q2',)]`.

            If just an iterable of labels is given, e.g. `('Q0','Q1')`, it is
            assumed to specify the first and only tensor product block.

            If a single state space label is given, e.g. `'Q2'`, then it is
            assumed to completely specify the first and only tensor product
            block.

        dims : int or iterable, optional
            The dimension of each state space label as an integer, tuple of
            integers, or list or tuples of integers to match the structure
            of `label_list` (i.e., if `label_list=('Q0','Q1')` then `dims` should
            be a tuple of 2 integers).  Values specify state-space dimensions: 2
            for a qubit, 3 for a qutrit, etc.  If None, then the dimensions are
            inferred, if possible, from the following naming rules:

            - if the label starts with 'L', dim=1 (a single Level)
            - if the label starts with 'Q' OR is an int, dim=2 (a Qubit)
            - if the label starts with 'T', dim=3 (a quTrit)

        types : str or iterable, optional
            A list of label types, either `'Q'` or `'C'` for "quantum" and
            "classical" respectively, indicating the type of state-space
            associated with each label.  Like `dims`, `types` must match
            the structure of `label_list`.  A quantum state space of dimension
            `d` is a `d`-by-`d` density matrix, whereas a classical state space
            of dimension d is a vector of `d` probabilities.  If `None`, then
            all labels are assumed to be quantum.

        evotype : {"densitymx","statevec","stabilizer","svterm","cterm"}
            The evolution type that this state-space will be used with.  This
            information is needed just to select the appropriate default
            dimensions, e.g. whether a qubit has a 2- or 4-dimensional state
            space.
        """

        #Allow initialization via another CustomStateSpace object
        if isinstance(label_list, CustomStateSpace):
            assert(dims is None and types is None), "Clobbering non-None 'dims' and/or 'types' arguments"
            dims = [tuple((label_list.labeldims[lbl] for lbl in tpbLbls))
                    for tpbLbls in label_list.labels]
            types = [tuple((label_list.labeltypes[lbl] for lbl in tpbLbls))
                     for tpbLbls in label_list.labels]
            label_list = label_list.labels

        #Step1: convert label_list (and dims, if given) to a list of
        # elements describing each "tensor product block" - each of
        # which is a tuple of string labels.

        def is_label(x):
            """ Return whether x is a valid space-label """
            return isinstance(x, str) or isinstance(x, _numbers.Integral)

        if is_label(label_list):
            label_list = [(label_list,)]
            if dims is not None: dims = [(dims,)]
            if types is not None: types = [(types,)]
        else:
            #label_list must be iterable if it's not a string
            label_list = list(label_list)

        if len(label_list) > 0 and is_label(label_list[0]):
            # assume we've just been give the labels for a single tensor-prod-block
            label_list = [label_list]
            if dims is not None: dims = [dims]
            if types is not None: types = [types]

        self.labels = tuple([tuple(tpbLabels) for tpbLabels in label_list])

        #Type check - labels must be strings or ints
        for tpbLabels in self.labels:  # loop over tensor-prod-blocks
            for lbl in tpbLabels:
                if not is_label(lbl):
                    raise ValueError("'%s' is an invalid state-space label (must be a string or integer)" % lbl)

        # Get the type of each labeled space
        self.labeltypes = {}
        if types is None:  # use defaults
            for tpbLabels in self.labels:  # loop over tensor-prod-blocks
                for lbl in tpbLabels:
                    self.labeltypes[lbl] = 'C' if (isinstance(lbl, str) and lbl.startswith('C')) else 'Q'  # default
        else:
            for tpbLabels, tpbTypes in zip(self.labels, types):
                for lbl, typ in zip(tpbLabels, tpbTypes):
                    self.labeltypes[lbl] = typ

        # Get the dimension of each labeled space
        self.labeldims = {}
        if dims is None:
            for tpbLabels in self.labels:  # loop over tensor-prod-blocks
                for lbl in tpbLabels:
                    if isinstance(lbl, _numbers.Integral): d = 2  # ints = qubits
                    elif lbl.startswith('T'): d = 3  # qutrit
                    elif lbl.startswith('Q'): d = 2  # qubits
                    elif lbl.startswith('L'): d = 1  # single level
                    elif lbl.startswith('C'): d = 2  # classical bits
                    else: raise ValueError("Cannot determine state-space dimension from '%s'" % lbl)
                    if evotype not in ('statevec', 'stabilizer', 'chp') and self.labeltypes[lbl] == 'Q':
                        d = d**2  # density-matrix spaces have squared dim
                        # ("densitymx","svterm","cterm") all use super-ops
                    self.labeldims[lbl] = d
        else:
            for tpbLabels, tpbDims in zip(self.labels, dims):
                for lbl, dim in zip(tpbLabels, tpbDims):
                    self.labeldims[lbl] = dim

        # Store the starting index (within the density matrix / state vec) of
        # each tensor-product-block (TPB), and which labels belong to which TPB
        self.tpb_index = {}

        self.tpb_dims = []
        for iTPB, tpbLabels in enumerate(self.labels):
            float_prod = _np.product(_np.array([self.labeldims[lbl] for lbl in tpbLabels], 'd'))
            if float_prod >= float(_sys.maxsize):  # too many qubits to hold dimension in an integer
                self.tpb_dims.append(_np.inf)
            else:
                self.tpb_dims.append(int(_np.product([self.labeldims[lbl] for lbl in tpbLabels])))
            self.tpb_index.update({lbl: iTPB for lbl in tpbLabels})

        self.dim = sum(self.tpb_dims)

        if len(self.labels) == 1 and all([v == 2 for v in self.labeldims.values()]):
            self.nqubits = len(self.labels[0])  # there's a well-defined number of qubits
        else:
            self.nqubits = None

    def reduce_dims_densitymx_to_state_inplace(self):
        """
        Reduce all state space dimensions appropriately for moving from a density-matrix to state-vector representation.

        Returns
        -------
        None
        """
        for lbl in self.labeldims:
            if self.labeltypes[lbl] == 'Q':
                self.labeldims[lbl] = int(_np.sqrt(self.labeldims[lbl]))

        #update tensor-product-block dims and overall dim too:
        self.tpb_dims = []
        for iTPB, tpbLabels in enumerate(self.labels):
            self.tpb_dims.append(int(_np.product([self.labeldims[lbl] for lbl in tpbLabels])))
            self.tpb_index.update({lbl: iTPB for lbl in tpbLabels})
        self.dim = sum(self.tpb_dims)

    def num_tensor_prod_blocks(self):
        """
        Get the number of tensor-product blocks which are direct-summed to get the final state space.

        Returns
        -------
        int
        """
        return len(self.labels)

    def tensor_product_block_labels(self, i_tpb):  # unused
        """
        Get the labels for the `iTBP`-th tensor-product block.

        Parameters
        ----------
        i_tpb : int
            Tensor-product block index.

        Returns
        -------
        tuple
        """
        return self.labels[i_tpb]

    def tensor_product_block_dims(self, i_tpb):  # unused
        """
        Get the dimension corresponding to each label in the `iTBP`-th tensor-product block.

        The dimension of the entire block is the product of these.

        Parameters
        ----------
        i_tpb : int
            Tensor-product block index.

        Returns
        -------
        tuple
        """
        return tuple((self.labeldims[lbl] for lbl in self.labels[i_tpb]))

    def product_dim(self, labels):  # only in modelconstruction
        """
        Computes the product of the state-space dimensions associated with each label in `labels`.

        Parameters
        ----------
        labels : list
            A list of state space labels (strings or integers).

        Returns
        -------
        int
        """
        return int(_np.product([self.labeldims[l] for l in labels]))

    def __str__(self):
        if len(self.labels) == 0: return "ZeroDimSpace"
        return ' + '.join(
            ['*'.join(["%s(%d%s)" % (lbl, self.labeldims[lbl], 'c' if (self.labeltypes[lbl] == 'C') else '')
                       for lbl in tpb]) for tpb in self.labels])

    def __repr__(self):
        return "CustomStateSpace[" + str(self) + "]"

    def copy(self):
        """
        Return a copy of this CustomStateSpace.

        Returns
        -------
        CustomStateSpace
        """
        return _copy.deepcopy(self)


def default_space_for_dim(dim):
    pass


def default_space_for_udim(dim):
    pass


def default_space_for_num_qubits(num_qubits):
    pass
