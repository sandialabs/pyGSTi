"""
Defines the MarginalizedPOVM class
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

#from .. import modelmember as _mm
from .povm import POVM as _POVM
from .staticeffect import StaticPOVMEffect as _StaticPOVMEffect
from ...models import labeldicts as _ld
from ...objects.label import Label as _Label


class MarginalizedPOVM(_POVM):
    """
    A POVM whose effects are the sums of sets of effect vectors in a parent POVM.

    Namely the effects of the parent POVN whose labels have the same *character*
    at certain "marginalized" indices are summed together.

    Parameters
    ----------
    povm_to_marginalize : POVM
        The POVM to marginalize (the "parent" POVM).

    all_sslbls : StateSpaceLabels or tuple
        The state space labels of the parent POVM, which should have as many
        labels (factors) as the parent POVM's outcome/effect labels have characters.

    sslbls_after_marginalizing : tuple
        The subset of `all_sslbls` that should be *kept* after marginalizing.
    """

    def __init__(self, povm_to_marginalize, all_sslbls, sslbls_after_marginalizing):
        """
        Create a MarginalizedPOVM.

        Create a marginalized POVM by adding together sets of effect vectors whose labels
        have the same *character* at marginalized indices.  This assumes that the POVM
        being marginalized has a particular (though common) effect-label structure whereby
        each state-space sector corresponds to a single character, e.g. "0010" for a 4-qubt POVM.

        Parameters
        ----------
        povm_to_marginalize : POVM
            The POVM to marginalize (the "parent" POVM).

        all_sslbls : StateSpaceLabels or tuple
            The state space labels of the parent POVM, which should have as many
            labels (factors) as the parent POVM's outcome/effect labels have characters.

        sslbls_after_marginalizing : tuple
            The subset of `all_sslbls` that should be *kept* after marginalizing.
        """
        self.povm_to_marginalize = povm_to_marginalize

        if isinstance(all_sslbls, _ld.StateSpaceLabels):
            assert(len(all_sslbls.labels) == 1), "all_sslbls should only have a single tensor product block!"
            all_sslbls = all_sslbls.labels[0]

        #now all_sslbls is a tuple of labels, like sslbls_after_marginalizing
        self.sslbls_to_marginalize = all_sslbls
        self.sslbls_after_marginalizing = sslbls_after_marginalizing
        indices_to_keep = set([list(all_sslbls).index(l) for l in sslbls_after_marginalizing])
        indices_to_remove = set(range(len(all_sslbls))) - indices_to_keep
        self.indices_to_marginalize = sorted(indices_to_remove, reverse=True)

        elements_to_sum = {}
        for k in self.povm_to_marginalize.keys():
            mk = self.marginalize_effect_label(k)
            if mk in elements_to_sum:
                elements_to_sum[mk].append(k)
            else:
                elements_to_sum[mk] = [k]
        self._elements_to_sum = {k: tuple(v) for k, v in elements_to_sum.items()}  # convert to tuples
        super(MarginalizedPOVM, self).__init__(self.povm_to_marginalize.dim, self.povm_to_marginalize._evotype)

    def marginalize_effect_label(self, elbl):
        """
        Removes the "marginalized" characters from `elbl`, resulting in a marginalized POVM effect label.

        Parameters
        ----------
        elbl : str
            Effect label (typically of the parent POVM) to marginalize.
        """
        assert(len(elbl) == len(self.sslbls_to_marginalize))
        for i in self.indices_to_marginalize:
            elbl = elbl[:i] + elbl[i + 1:]  # remove i-th character
        return elbl

    def __contains__(self, key):
        """ For lazy creation of effect vectors """
        return bool(key in self._elements_to_sum)

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return len(self._elements_to_sum)

    def keys(self):
        """
        An iterator over the effect (outcome) labels of this POVM.
        """
        for k in self._elements_to_sum.keys():
            yield k

    def values(self):
        """
        An iterator over the effect SPAM vectors of this POVM.
        """
        for k in self.keys():
            yield self[k]

    def items(self):
        """
        An iterator over the (effect_label, effect_vector) items in this POVM.
        """
        for k in self.keys():
            yield k, self[k]

    def __getitem__(self, key):
        """ For lazy creation of effect vectors """
        if _collections.OrderedDict.__contains__(self, key):
            return _collections.OrderedDict.__getitem__(self, key)
        elif key in self:  # calls __contains__ to efficiently check for membership
            #create effect vector now that it's been requested (lazy creation)
            #FUTURE: maybe have a "SumSPAMVec" that can add spamvecs to preserve paramterization and avoid dense reps
            effect_vec = None  # Note: currently all marginalized POVMs are *static*, since
            # we don't have a good general way to add parameterized effect vectors.

            for k in self._elements_to_sum[key]:
                e = self.povm_to_marginalize[k]
                if effect_vec is None:
                    effect_vec = e.to_dense()
                else:
                    effect_vec += e.to_dense()
            effect = _StaticPOVMEffect(effect_vec, self._evotype)
            effect.set_gpindices(slice(0, 0), self.parent)
            _collections.OrderedDict.__setitem__(self, key, effect)
            return effect
        else: raise KeyError("%s is not an outcome label of this MarginalizedPOVM" % key)

    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        return (MarginalizedPOVM, (self.povm_to_marginalize, self.sslbls_to_marginalize,
                                   self.sslbls_after_marginalizing),
                {'_gpindices': self._gpindices})  # preserve gpindices (but not parent)

    #May need to implement this in future if we allow non-static MarginalizedPOVMs
    #def allocate_gpindices(self, starting_index, parent, memo=None):
    #    """
    #    Sets gpindices array for this object or any objects it
    #    contains (i.e. depends upon).  Indices may be obtained
    #    from contained objects which have already been initialized
    #    (e.g. if a contained object is shared with other
    #     top-level objects), or given new indices starting with
    #    `starting_index`.
    #
    #    Parameters
    #    ----------
    #    starting_index : int
    #        The starting index for un-allocated parameters.
    #
    #    parent : Model or ModelMember
    #        The parent whose parameter array gpindices references.
    #
    #    memo : set, optional
    #        Used to prevent duplicate calls and self-referencing loops.  If
    #        `memo` contains an object's id (`id(self)`) then this routine
    #        will exit immediately.
    #
    #    Returns
    #    -------
    #    num_new: int
    #        The number of *new* allocated parameters (so
    #        the parent should mark as allocated parameter
    #        indices `starting_index` to `starting_index + new_new`).
    #    """
    #    if memo is None: memo = set()
    #    if id(self) in memo: return 0
    #    memo.add(id(self))
    #
    #    assert(self.base_povm.num_params == 0)  # so no need to do anything w/base_povm
    #    num_new_params = self.error_map.allocate_gpindices(starting_index, parent, memo)  # *same* parent as self
    #    _mm.ModelMember.set_gpindices(
    #        self, self.error_map.gpindices, parent)
    #    return num_new_params

    #def submembers(self):
    #    """
    #    Get the ModelMember-derived objects contained in this one.
    #
    #    Returns
    #    -------
    #    list
    #    """
    #    return [self.povm_to_marginalize]
    #
    #def relink_parent(self, parent):  # Unnecessary?
    #    """
    #    Sets the parent of this object *without* altering its gpindices.
    #
    #    In addition to setting the parent of this object, this method
    #    sets the parent of any objects this object contains (i.e.
    #    depends upon) - much like allocate_gpindices.  To ensure a valid
    #    parent is not overwritten, the existing parent *must be None*
    #    prior to this call.
    #    """
    #    self.povm_to_marginalize.relink_parent(parent)
    #    _mm.ModelMember.relink_parent(self, parent)

    #def set_gpindices(self, gpindices, parent, memo=None):
    #    """
    #    Set the parent and indices into the parent's parameter vector that
    #    are used by this ModelMember object.
    #
    #    Parameters
    #    ----------
    #    gpindices : slice or integer ndarray
    #        The indices of this objects parameters in its parent's array.
    #
    #    parent : Model or ModelMember
    #        The parent whose parameter array gpindices references.
    #
    #    Returns
    #    -------
    #    None
    #    """
    #    if memo is None: memo = set()
    #    elif id(self) in memo: return
    #    memo.add(id(self))
    #
    #    assert(self.base_povm.num_params == 0)  # so no need to do anything w/base_povm
    #    self.error_map.set_gpindices(gpindices, parent, memo)
    #    self.terms = {}  # clear terms cache since param indices have changed now
    #    _mm.ModelMember._set_only_my_gpindices(self, gpindices, parent)

    def simplify_effects(self, prefix=""):
        """
        Creates a dictionary of simplified effect vectors.

        Returns a dictionary of effect SPAMVecs that belong to the POVM's parent
        `Model` - that is, whose `gpindices` are set to all or a subset of
        this POVM's gpindices.  Such effect vectors are used internally within
        computations involving the parent `Model`.

        Parameters
        ----------
        prefix : str
            A string, usually identitying this POVM, which may be used
            to prefix the simplified gate keys.

        Returns
        -------
        OrderedDict of SPAMVecs
        """
        # Create "simplified" effect vectors, which infer their parent and
        # gpindices from the set of "factor-POVMs" they're constructed with.
        if isinstance(prefix, _Label):  # Deal with case when prefix isn't just a string
            simplified = _collections.OrderedDict(
                [(_Label(prefix.name + '_' + k, prefix.sslbls), self[k]) for k in self.keys()])
        else:
            if prefix: prefix += "_"
            simplified = _collections.OrderedDict(
                [(prefix + k, self[k]) for k in self.keys()])
        return simplified

    def __str__(self):
        s = "Marginalized POVM of length %d\n" \
            % (len(self))
        return s
