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
import collections as _collections
import numpy as _np
import copy as _copy
import numbers as _numbers
import warnings as _warnings

from . import spamvec as _sv
from . import operation as _op
from . import modelmember as _gm
from .label import Label as _Label


class _PrefixOrderedDict(_collections.OrderedDict):
    """
    Base class ordered dictionaries whose keys *must* be strings which begin with a given prefix.

    Parameters
    ----------
    prefix : str
        The required prefix.

    items : list or dict, optional
        Initial values.  Should only be used as part of de-serialization.
    """

    def __init__(self, prefix, items=[]):
        """ Creates a new _PrefixOrderedDict whose keys must begin
            with the string `prefix`."""
        #** Note: if change __init__ signature, update __reduce__ below
        self._prefix = prefix
        super(_PrefixOrderedDict, self).__init__(items)

    def __setitem__(self, key, val):
        """ Assumes key is a Label object """
        if not (self._prefix is None or key.has_prefix(self._prefix)):
            raise KeyError("All keys must be strings, "
                           "beginning with the prefix '%s'" % self._prefix)
        super(_PrefixOrderedDict, self).__setitem__(key, val)

    #Handled by derived classes
    #def __reduce__(self):
    #    items = [(k,v) for k,v in self.iteritems()]
    #    return (_PrefixOrderedDict, (self._prefix, items), None)

    """
    An ordered dictionary whose keys must begin with a given prefix,
    and which holds LinearOperator objects.  This class ensures that every value is a
    :class:`LinearOperator`-derived object by converting any non-`LinearOperator` values into
    `LinearOperator`s upon assignment and raising an error if this is not possible.
    """


class OrderedMemberDict(_PrefixOrderedDict, _gm.ModelChild):
    """
    An ordered dictionary whose keys must begin with a given prefix.

    This class also ensure that every value is an object of the appropriate Model
    member type (e.g. :class:`SPAMVec`- or :class:`LinearOperator`-derived object) by converting any
    values into that type upon assignment and raising an error if this is not possible.

    Parameters
    ----------
    parent : Model
        The parent model, needed to obtain the dimension and handle
        updates to parameters.

    default_param : {"TP","full",...}
        The default parameterization used when creating an
        object from a key assignment.

    prefix : str
        The required prefix of all keys (which must be strings).

    flags : dict
        A dictionary of flags adjusting the behavior of the created
        object.  Allowed keys are:

        - `'cast_to_type'`: {`"operation"`,`"spamvec"`,`None`} -- whether
          (or not) to automatically convert assigned values to a particular
          type of `ModelMember` object. (default is `None`)
        - `'auto_embed'` : bool -- whether or not to automatically embed
          objects with a lower dimension than this `OrderedMemberDict`'s
          parent model. (default is `False`).
        - `'match_parent_dim'` : bool -- whether or not to require that
          all contained objects match the parent `Model`'s dimension
          (perhaps after embedding).  (default is `False`)
        - `'match_parent_evotype'` : bool -- whether or not to require that
          all contained objects match the parent `Model`'s evolution type.
          (default is `False`).

    items : list, optional
        Used by pickle and other serializations to initialize elements.
    """

    def __init__(self, parent, default_param, prefix, flags, items=[]):
        """
        Creates a new OrderedMemberDict.

        Parameters
        ----------
        parent : Model
            The parent model, needed to obtain the dimension and handle
            updates to parameters.

        default_param : {"TP","full",...}
            The default parameterization used when creating an
            object from a key assignment.

        prefix : str
            The required prefix of all keys (which must be strings).

        flags : dict
            A dictionary of flags adjusting the behavior of the created
            object.  Allowed keys are:

            - `'cast_to_type'`: {`"operation"`,`"spamvec"`,`None`} -- whether
              (or not) to automatically convert assigned values to a particular
              type of `ModelMember` object. (default is `None`)
            - `'auto_embed'` : bool -- whether or not to automatically embed
              objects with a lower dimension than this `OrderedMemberDict`'s
              parent model. (default is `False`).
            - `'match_parent_dim'` : bool -- whether or not to require that
              all contained objects match the parent `Model`'s dimension
              (perhaps after embedding).  (default is `False`)
            - `'match_parent_evotype'` : bool -- whether or not to require that
              all contained objects match the parent `Model`'s evolution type.
              (default is `False`).

        items : list, optional
            Used by pickle and other serializations to initialize elements.
        """
        #** Note: if change __init__ signature, update __reduce__ below
        if isinstance(flags, str):  # for backward compatibility
            flags = {'cast_to_type': ("operation" if flags == "gate" else flags)}

        # Note: we *don't* want to be calling parent's "rebuild" function here,
        # when we're creating a new list, as this behavior is only intended for
        # explicit insertions.  Since calling the base class __init__ will
        # call this class's __setitem__ we set parent to None for this step.
        self.parent = None  # so __init__ below doesn't call _rebuild_paramvec
        self.default_param = default_param  # "TP", "full", "static", "clifford",...
        self.flags = {'auto_embed': flags.get('auto_embed', False),
                      'match_parent_dim': flags.get('match_parent_dim', False),
                      'match_parent_evotype': flags.get('match_parent_evotype', False),
                      'cast_to_type': flags.get('cast_to_type', None)
                      }
        _PrefixOrderedDict.__init__(self, prefix, items)
        _gm.ModelChild.__init__(self, parent)  # set's self.parent

        #Set parent our elements, now that the list has been initialized
        # (done for un-pickling b/c reduce => __init__ is called to construct
        #  unpickled object)
        if self.parent is not None:
            for el in self.values(): el.set_gpindices(el.gpindices, self.parent)
            #sets parent and retains any existing indices in elements

    def _check_dim(self, obj):
        if not self.flags['match_parent_dim']: return  # no check
        if isinstance(obj, _gm.ModelMember):
            dim = obj.dim
        elif self.flags['cast_to_type'] == "spamvec":
            dim = len(obj)
        elif self.flags['cast_to_type'] == "operation":
            try:
                d1 = len(obj)
                len(obj[0])
                # XXX this is an abuse of exception handling
            except:
                raise ValueError("%s doesn't look like a 2D array/list" % obj)
            if any([len(row) != d1 for row in obj]):
                raise ValueError("%s is not a *square* 2D array" % obj)
            dim = d1
        else:
            raise ValueError("Cannot obtain dimension!")

        if self.parent is None: return
        elif self.parent.dim != dim:
            raise ValueError("Cannot add object with dimension "
                             "%s to model of dimension %d"
                             % (dim, self.parent.dim))

    def _check_evotype(self, evotype):
        if not self.flags['match_parent_evotype']: return  # no check
        if self.parent is None: return
        elif self.parent._evotype != evotype:
            raise ValueError(("Cannot add an object with evolution type"
                              " '%s' to a model with one of '%s'") %
                             (evotype, self.parent._evotype))

    def __contains__(self, key):
        if not isinstance(key, _Label): key = _Label(key, None)
        return super(OrderedMemberDict, self).__contains__(key)

    def __getitem__(self, key):
        #if self.parent is not None:
        #    #print("DEBUG: cleaning paramvec before getting ", key)
        #    self.parent._clean_paramvec()
        if not isinstance(key, _Label): key = _Label(key, None)
        return super(OrderedMemberDict, self).__getitem__(key)

    def _auto_embed(self, key_label, value):
        if not self.flags['auto_embed']: return value  # no auto-embedding
        if self.parent is not None and key_label.sslbls is not None:
            parent_sslbls = self.parent.state_space_labels
            parent_sslbls = parent_sslbls.labels[0] if len(parent_sslbls.labels) == 1 else None  # only 1st TPB
            if parent_sslbls == key_label.sslbls: return value  # no need to embed, as key_label uses *all* sslbls

            if self.flags['cast_to_type'] == "operation":
                return self.parent._embed_operation(key_label.sslbls, self.cast_to_model_member(value))
            else:
                raise NotImplementedError("Cannot auto-embed objects other than opeations yet (not %s)."
                                          % self.flags['cast_to_type'])
        else:
            return value

    def cast_to_model_member(self, value):
        """
        Cast `value` to an object with the default parameterization if it's not a :class:`ModelMember`.

        Creates an object from `value` with a type given by the default
        parameterization if `value` isn't a :class:`ModelMember`.

        Parameters
        ----------
        value : object
            The object to act on.

        Returns
        -------
        object
        """
        if isinstance(value, _gm.ModelMember): return value
        if self.flags['cast_to_type'] is None:
            raise ValueError("Can only assign `ModelMember` objects as values (not %s)."
                             % str(type(value)))

        basis = self.parent.basis if self.parent else None
        obj = None
        if self.flags['cast_to_type'] == "spamvec":
            obj = _sv.StaticSPAMVec(value)
            obj = _sv.convert(obj, self.default_param, basis)
        elif self.flags['cast_to_type'] == "operation":
            obj = _op.StaticDenseOp(value)
            obj = _op.convert(obj, self.default_param, basis)
        return obj

    def __setitem__(self, key, value):
        if not isinstance(key, _Label): key = _Label(key)
        value = self._auto_embed(key, value)  # automatically create an embedded gate if needed
        self._check_dim(value)

        if isinstance(value, _gm.ModelMember):  # if we're given an object, just replace
            #When self has a valid parent (as it usually does, except when first initializing)
            # we copy and reset the gpindices & parent of ModelMember values which either:
            # 1) belong to a different parent (indices would be inapplicable if they exist)
            # 2) have indices but no parent (indices are inapplicable to us)
            # Note that we don't copy and reset the case when a value's parent and gpindices
            #  are both None, as gpindices==None indicates that the value may not have had
            #  its gpindices allocated yet and so *might* have "latent" gpindices that do
            #  belong to our parent (self.parent) (and copying a value will reset all
            #  the parents to None).
            wrongParent = (value.parent is not None) and (value.parent is not self.parent)
            inappInds = (value.parent is None) and (value.gpindices is not None)

            if self.parent is not None and (wrongParent or inappInds):
                value = value.copy()  # copy value (so we don't mess up other parent) and
                value.set_gpindices(None, self.parent)  # erase gindices don't apply to us

            if not hasattr(value, "_evotype"): value._evotype = "densitymx"  # for backward compatibility
            self._check_evotype(value._evotype)

            if self.parent is not None and key in self:
                existing = super(OrderedMemberDict, self).__getitem__(key)
            else: existing = None

            if self.parent is not None: value.set_gpindices(None, self.parent)  # set parent
            super(OrderedMemberDict, self).__setitem__(key, value)

            # let the now-replaced existing object know it's been
            # removed from the parent, allowing it to reset (to None)
            # its parent link if there are no more references to it.
            if existing is not None and value is not existing:
                assert(existing.parent is None or existing.parent is self.parent), "Model object not setup correctly"
                existing.unlink_parent()

        elif key in self:  # if a object already exists...
            #try to set its value
            super(OrderedMemberDict, self).__getitem__(key).set_dense(value)

        else:
            #otherwise, we've been given a non-ModelMember-object that doesn't
            # exist yet, so use default creation flags to make one:
            obj = self.cast_to_model_member(value)

            if obj is None:
                raise ValueError("Cannot set a value of type: ", type(value))

            self._check_evotype(obj._evotype)
            if self.parent is not None: obj.set_gpindices(None, self.parent)
            super(OrderedMemberDict, self).__setitem__(key, obj)

        #rebuild Model's parameter vector (params may need to be added)
        if self.parent is not None:
            #print("DEBUG: marking paramvec for rebuild after inserting ", key, " : ", list(self.keys()))
            self.parent._mark_for_rebuild(super(OrderedMemberDict, self).__getitem__(key))
            # mark the parent's (Model's) paramvec for rebuilding

    def __delitem__(self, key):
        """Implements `del self[key]`"""
        if not isinstance(key, _Label): key = _Label(key, None)
        super(OrderedMemberDict, self).__delitem__(key)
        if self.parent is not None:
            #print("DEBUG: rebuilding paramvec after deleting ", key, " : ", list(self.keys()))
            self.parent._rebuild_paramvec()

    def copy(self, parent=None):
        """
        Returns a copy of this OrderedMemberDict.

        Parameters
        ----------
        parent : Model
            The new parent Model, if one exists.  Typically, when copying
            an OrderedMemberDict you want to reset the parent.

        Returns
        -------
        OrderedMemberDict
        """
        return OrderedMemberDict(parent, self.default_param,
                                 self._prefix, self.flags,
                                 [(lbl, val.copy(parent)) for lbl, val in self.items()])

    def __reduce__(self):
        #Call constructor to create object, but with parent == None to avoid
        # circular pickling of Models.  Must set parent separately.
        return (OrderedMemberDict,
                (None, self.default_param, self._prefix, self.flags, list(self.items())), None)

    def __pygsti_reduce__(self):
        return self.__reduce__()


class OutcomeLabelDict(_collections.OrderedDict):
    """
    An ordered dictionary of outcome labels, whose keys are tuple-valued outcome labels.

    This class extends an ordinary OrderedDict by implements mapping
    string-values single-outcome labels to 1-tuples containing that
    label (and vice versa), allowing the use of strings as outcomes
    labels from the user's perspective.

    Parameters
    ----------
    items : list or dict, optional
        Initial values.  Should only be used as part of de-serialization.

    Attributes
    ----------
    _strict : bool
        Whether mapping from strings to 1-tuples is performed.
    """

    #Whether mapping from strings to 1-tuples is performed
    _strict = False

    @classmethod
    def to_outcome(cls, val):
        """
        Converts string outcomes like "0" to proper outcome tuples, like ("0",).

        (also converts non-tuples to tuples, e.g. `["0","1"]` to `("0","1")` )

        Parameters
        ----------
        val : str or tuple
            The value to convert into an outcome label (i.e. a tuple)

        Returns
        -------
        tuple
        """
        return (val,) if isinstance(val, str) else tuple(val)

    def __init__(self, items=[]):
        """
        Creates a new OutcomeLabelDict.

        Parameters
        ----------
        items : list, optional
            Used by pickle and other serializations to initialize elements.
        """
        #** Note: if change __init__ signature, update __reduce__ below
        super(OutcomeLabelDict, self).__init__(items)

    def __getitem__(self, key):
        if not OutcomeLabelDict._strict:
            key = OutcomeLabelDict.to_outcome(key)
        return super(OutcomeLabelDict, self).__getitem__(key)

    def __setitem__(self, key, val):
        if not OutcomeLabelDict._strict:
            key = OutcomeLabelDict.to_outcome(key)
        super(OutcomeLabelDict, self).__setitem__(key, val)

    def getitem_unsafe(self, key, defaultval):
        """
        Gets an item without checking that `key` is a properly formatted outcome tuple.

        Only use this method when you're sure `key` is an outcome tuple and not, e.g.,
        just a string.

        Parameters
        ----------
        key : object
            The key to retrieve

        defaultval : object
            The default value to use (if the key is absent).

        Returns
        -------
        object
        """
        return super(OutcomeLabelDict, self).get(key, defaultval)

    def setitem_unsafe(self, key, val):
        """
        Sets item without checking that the key is a properly formatted outcome tuple.

        Only use this method when you're sure `key` is an outcome tuple and not, e.g.,
        just a string.

        Parameters
        ----------
        key : object
            The key to retrieve.

        val : object
            the value to set.

        Returns
        -------
        None
        """
        super(OutcomeLabelDict, self).__setitem__(key, val)

    def __contains__(self, key):
        if not OutcomeLabelDict._strict:
            key = OutcomeLabelDict.to_outcome(key)
        return key in super(OutcomeLabelDict, self).keys()

    def contains_unsafe(self, key):
        """
        Checks for `key` without ensuring that it is a properly formatted outcome tuple.

        Only use this method when you're sure `key` is an outcome tuple and not, e.g.,
        just a string.

        Parameters
        ----------
        key : object
            The key to retrieve.

        Returns
        -------
        bool
        """
        return super(OutcomeLabelDict, self).__contains__(key)

    def copy(self):
        """
        Return a copy of this OutcomeLabelDict.

        Returns
        -------
        OutcomeLabelDict
        """
        return OutcomeLabelDict([(lbl, _copy.deepcopy(val))
                                 for lbl, val in self.items()])

    def __pygsti_reduce__(self):
        items = [(k, v) for k, v in self.items()]
        return (OutcomeLabelDict, (items,), None)

    def __reduce__(self):
        items = [(k, v) for k, v in self.items()]
        return (OutcomeLabelDict, (items,), None)


class StateSpaceLabels(object):
    """
    A labeling of the decomposition of a state space (Hilbert Space).

    A StateSpaceLabels object describes, using string/int labels, how an entire
    Hilbert state space is decomposed into the direct sum of terms which
    themselves are tensor products of smaller (typically qubit-sized) Hilber
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

    def __init__(self, label_list, dims=None, types=None, evotype="densitymx"):
        """
        Creates a new StateSpaceLabels object.

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

        #Allow initialization via another StateSpaceLabels object
        if isinstance(label_list, StateSpaceLabels):
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
                    if evotype not in ('statevec', 'stabilizer') and self.labeltypes[lbl] == 'Q':
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
            self.tpb_dims.append(int(_np.product([self.labeldims[lbl] for lbl in tpbLabels])))
            self.tpb_index.update({lbl: iTPB for lbl in tpbLabels})

        self.dim = sum(self.tpb_dims)

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

    def num_tensor_prod_blocks(self):  # only in modelconstruction.py
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
        return "StateSpaceLabels[" + str(self) + "]"

    def copy(self):
        """
        Return a copy of this StateSpaceLabels.

        Returns
        -------
        StateSpaceLabels
        """
        return _copy.deepcopy(self)
