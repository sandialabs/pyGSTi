"""
Defines OrderedDict-derived classes used to store specific pyGSTi objects
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
import collections as _collections
import copy as _copy

from pygsti.baseobjs.label import Label as _Label
from pygsti.modelmembers import modelmember as _mm


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

    def __init__(self, prefix, items=None):
        """ Creates a new _PrefixOrderedDict whose keys must begin
            with the string `prefix`."""
        #** Note: if change __init__ signature, update __reduce__ below
        if items is None:
            items = []
        self._prefix = prefix
        super(_PrefixOrderedDict, self).__init__(items)

    def __setitem__(self, key, val):
        """ Assumes key is a Label object """
        if not (self._prefix is None or key.has_prefix(self._prefix)):
            raise KeyError("All keys must be strings, "
                           "beginning with the prefix '%s'" % self._prefix)
        super(_PrefixOrderedDict, self).__setitem__(key, val)


class OrderedMemberDict(_PrefixOrderedDict, _mm.ModelChild):
    """
    An ordered dictionary whose keys must begin with a given prefix.

    This class also ensure that every value is an object of the appropriate Model
    member type (e.g. :class:`State`- or :class:`LinearOperator`-derived object) by converting any
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

        - `'cast_to_type'`: {`"operation"`,`"state"`,`None`} -- whether
          (or not) to automatically convert assigned values to a particular
          type of `ModelMember` object. (default is `None`)
        - `'auto_embed'` : bool -- whether or not to automatically embed
          objects with a lower dimension than this `OrderedMemberDict`'s
          parent model. (default is `False`).
        - `'match_parent_statespace'` : bool -- whether or not to require that
          all contained objects match the parent `Model`'s state space
          (perhaps after embedding).  (default is `False`)
        - `'match_parent_evotype'` : bool -- whether or not to require that
          all contained objects match the parent `Model`'s evolution type.
          (default is `False`).

    items : list, optional
        Used by pickle and other serializations to initialize elements.
    """

    def __init__(self, parent, default_param, prefix, flags, items=None):
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

            - `'cast_to_type'`: {`"operation"`,`"state"`,`None`} -- whether
              (or not) to automatically convert assigned values to a particular
              type of `ModelMember` object. (default is `None`)
            - `'auto_embed'` : bool -- whether or not to automatically embed
              objects with a lower dimension than this `OrderedMemberDict`'s
              parent model. (default is `False`).
            - `'match_parent_statespace'` : bool -- whether or not to require that
              all contained objects match the parent `Model`'s state space
              (perhaps after embedding).  (default is `False`)
            - `'match_parent_evotype'` : bool -- whether or not to require that
              all contained objects match the parent `Model`'s evolution type.
              (default is `False`).

        items : list, optional
            Used by pickle and other serializations to initialize elements.
        """
        #** Note: if change __init__ signature, update __reduce__ below
        if items is None:
            items = []
        if isinstance(flags, str):  # for backward compatibility
            flags = {'cast_to_type': ("operation" if flags == "gate" else flags)}

        # Note: we *don't* want to be calling parent's "rebuild" function here,
        # when we're creating a new list, as this behavior is only intended for
        # explicit insertions.  Since calling the base class __init__ will
        # call this class's __setitem__ we set parent to None for this step.
        self.parent = None  # so __init__ below doesn't call _rebuild_paramvec
        self.default_param = default_param  # "TP", "full", "static", "clifford",...
        self.flags = {'auto_embed': flags.get('auto_embed', False),
                      'match_parent_statespace': flags.get('match_parent_statespace', False),
                      'match_parent_evotype': flags.get('match_parent_evotype', False),
                      'cast_to_type': flags.get('cast_to_type', None)
                      }
        _PrefixOrderedDict.__init__(self, prefix, items)
        _mm.ModelChild.__init__(self, parent)  # set's self.parent

        #Set parent our elements, now that the list has been initialized
        # (done for un-pickling b/c reduce => __init__ is called to construct
        #  unpickled object)
        if self.parent is not None:
            for el in self.values(): el.set_gpindices(el.gpindices, self.parent)
            #sets parent and retains any existing indices in elements

    def _check_state_space(self, obj):
        if not self.flags['match_parent_statespace']: return  # no check
        if self.parent is None: return  # no check

        if isinstance(obj, _mm.ModelMember):
            if not self.parent.state_space.is_compatible_with(obj.state_space):
                raise ValueError("Cannot add object with state space "
                                 "%s to model with state space %s"
                                 % (str(obj.state_space), str(self.parent.state_space)))
        else:
            #Compare dimensions of vector/matrix-like object
            if self.flags['cast_to_type'] == "state":
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

            if self.parent.state_space.dim != dim:
                raise ValueError("Cannot add object with dimension "
                                 "%s to model of dimension %d"
                                 % (dim, self.parent.state_space.dim))

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
            parent_sslbls = self.parent.state_space
            parent_sslbls = parent_sslbls.tensor_product_block_labels(0) \
                if parent_sslbls.num_tensor_product_blocks == 1 else None  # only 1st TPB
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
        if isinstance(value, _mm.ModelMember): return value
        if self.flags['cast_to_type'] is None:
            raise ValueError("Can only assign `ModelMember` objects as *new* values (not %s)."
                             % str(type(value)))

        if self.parent is None:
            return None  # cannot cast to a model member without a parent model, since we need to know the evotype

        evotype = self.parent.evotype
        basis = self.parent.basis
        state_space = self.parent.state_space
        obj = None

        from ..modelmembers import states as _state
        from ..modelmembers import operations as _op
        from ..modelmembers import instruments as _inst
        from ..modelmembers import povms as _povm
        if self.flags['cast_to_type'] == "state":
            obj = _state.StaticState(value, basis, evotype, state_space)
            obj = _state.convert(obj, self.default_param, basis)
        elif self.flags['cast_to_type'] == "operation":
            obj = _op.StaticArbitraryOp(value, None, evotype, state_space)
            obj = _op.convert(obj, self.default_param, basis)
        elif self.flags['cast_to_type'] == "povm":
            obj = _povm.UnconstrainedPOVM(
                [_povm.StaticPOVMEffect(v, basis, evotype, state_space) for v in value],
                evotype, state_space)
            obj = _povm.convert(obj, self.default_param, basis)
        elif self.flags['cast_to_type'] == "instrument":
            members = []
            for v in value:
                m = _op.StaticArbitraryOp(v, None, evotype, state_space)
                members.append(_op.convert(m, self.default_param, basis))
            obj = _inst.Instrument(members, evotype, state_space)

        return obj

    def __setitem__(self, key, value):
        if not isinstance(key, _Label): key = _Label(key)
        value = self._auto_embed(key, value)  # automatically create an embedded gate if needed
        self._check_state_space(value)

        if isinstance(value, _mm.ModelMember):  # if we're given an object, just replace
            #When self has a valid parent (as it usually does, except when first initializing)
            # we copy and reset the gpindices & parent of ModelMember values *if* they
            # belong to a different parent (indices would be inapplicable if they exist).
            #
            # If a ModelMember's parent is None then we leave the indices alone, as they may
            # have been initialized to allow the object to fully function in the absence of a
            # parent model.  Still, these indices are inapplicable to us and will prompt a call
            # to value.allocate_gpindices(...) the next time the Model's parameter vector is rebuilt.
            #
            # Alternatively, gpindices==None indicates that the value may not have had
            #  its gpindices allocated yet and so *might* have "latent" (i.e. from-submember) gpindices
            #  that do belong to our parent (self.parent) (and copying a value will reset all
            #  the parents to None).
            #
            # Either way, we should only copy the value when its parent is non-None and not our parent.
            wrongParent = (value.parent is not None) and (value.parent is not self.parent)

            if self.parent is not None and wrongParent:
                value = value.copy()  # copy value (so we don't mess up other parent) and
                value.set_gpindices(None, self.parent)  # erase gindices that don't apply to us

            if not hasattr(value, "_evotype"): value._evotype = "densitymx"  # for backward compatibility
            self._check_evotype(value._evotype)

            if self.parent is not None and key in self:
                existing = super(OrderedMemberDict, self).__getitem__(key)
            else: existing = None

            super(OrderedMemberDict, self).__setitem__(key, value)
            new_item = value  # keep track of newly set item for later

            # let the now-replaced existing object know it's been
            # removed from the parent, allowing it to reset (to None)
            # its parent link if there are no more references to it.
            if existing is not None and value is not existing:
                assert(existing.parent is None or existing.parent is self.parent), "Model object not setup correctly"
                existing.unlink_parent()

        elif key in self:  # if a object already exists...
            #try to set its value
            super(OrderedMemberDict, self).__getitem__(key).set_dense(value)
            new_item = None  # keep track of newly set item for later

        else:
            #otherwise, we've been given a non-ModelMember-object that doesn't
            # exist yet, so use default creation flags to make one:
            obj = self.cast_to_model_member(value)

            if obj is None:
                raise ValueError("Cannot set a value of type: ", type(value))

            self._check_evotype(obj._evotype)
            if self.parent is not None: obj.set_gpindices(None, self.parent)
            super(OrderedMemberDict, self).__setitem__(key, obj)
            new_item = obj  # keep track of newly set item for later

        #rebuild Model's parameter vector is a new modelmember has been added (*number* of params may need to change)
        if new_item is not None and self.parent is not None:
            #print("DEBUG: marking paramvec for rebuild after inserting ", key, " : ", list(self.keys()))
            # mark the parent's (Model's) paramvec for rebuilding:
            self.parent._mark_for_rebuild(new_item)
            if new_item.parent is not self.parent:  # de-allocate any items allocated to other models
                new_item.unlink_parent(force=True)

    def __delitem__(self, key):
        """Implements `del self[key]`"""
        if not isinstance(key, _Label): key = _Label(key, None)
        super(OrderedMemberDict, self).__delitem__(key)
        if self.parent is not None:
            #print("DEBUG: rebuilding paramvec after deleting ", key, " : ", list(self.keys()))
            self.parent._rebuild_paramvec()

    def copy(self, parent=None, memo=None):
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
                                 [(lbl, val.copy(parent, memo)) for lbl, val in self.items()])

    def __reduce__(self):
        #Call constructor to create object, but with parent == None to avoid
        # circular pickling of Models.  Must set parent separately.
        return (OrderedMemberDict,
                (None, self.default_param, self._prefix, self.flags, list(self.items())), None)

    def __pygsti_reduce__(self):
        return self.__reduce__()
