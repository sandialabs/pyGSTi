"""
The EmbeddedInst class and supporting functionality.
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
import itertools as _itertools

import numpy as _np

from pygsti.modelmembers import modelmember as _modelmember
from pygsti.baseobjs.statespace import StateSpace as _StateSpace
from pygsti.modelmembers.operations import EmbeddedOp as _eop
from pygsti.modelmembers.instruments import Instrument as _Instrument


class EmbeddedInst(_Instrument):
    """
    An instrument containing a single lower (or equal) dimensional instrument within it.

    An EmbeddedInst acts as the identity on all of its domain except the
    subspace of its contained instrument, where it acts as the contained instrument does.

    Parameters
    ----------
    state_space : StateSpace
        Specifies the density matrix space upon which this instrument acts.

    target_labels : list of strs
        The labels contained in `state_space` which demarcate the
        portions of the state space acted on by `instrument_to_embed` (the
        "contained" instrument).

    instrument_to_embed : Instrument
        The instrument object that is to be contained within this instrument, and
        that specifies the only non-trivial action of the EmbeddedOp.
    """

    def __init__(self, state_space, target_labels, instrument_to_embed, allocated_to_parent=None):
        self.target_labels = tuple(target_labels) if (target_labels is not None) else None
        self.embedded_inst = instrument_to_embed

        assert(_StateSpace.cast(state_space).contains_labels(target_labels)), \
            "`target_labels` (%s) not found in `state_space` (%s)" % (str(target_labels), str(state_space))
        assert(self.embedded_inst.state_space.num_tensor_product_blocks == 1), \
            "EmbeddedInst objects can only embed instruments whose state spaces contain just a single tensor product block"
        assert(len(self.embedded_inst.state_space.sole_tensor_product_block_labels) == len(target_labels)), \
            "Embedded instrument's state space has a different number of components than the number of target labels!"

        member_ops = {}
        for outcome_label, gate in instrument_to_embed.items():
            new_outcome_label = str(outcome_label)
            for qubit_label in target_labels:
                new_outcome_label += ':' + str(qubit_label)      
            member_ops[new_outcome_label] = _eop(state_space, target_labels, gate, allocated_to_parent=allocated_to_parent)

        _Instrument.__init__(self, member_ops)
        self.init_gpindices(allocated_to_parent)

    def __getstate__(self):
        # Don't pickle 'instancemethod' or parent (see modelmember implementation)
        return _modelmember.ModelMember.__getstate__(self)

    def __setstate__(self, d):
        if "dirty" in d:  # backward compat: .dirty was replaced with ._dirty in ModelMember
            d['_dirty'] = d['dirty']; del d['dirty']
        self.__dict__.update(d)
        
    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return list(self.values())

    @property
    def parameter_labels(self):
        """
        An array of labels (usually strings) describing this model member's parameters.
        """
        return self.embedded_inst.parameter_labels

    @property
    def num_elements(self):
        """
        Return the number of total gate elements in this instrument.

        This is in general different from the number of *parameters*,
        which are the number of free variables used to generate all of
        the matrix *elements*.

        Returns
        -------
        int
        """
        return self.embedded_inst.num_elements

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this operation.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return self.embedded_inst.num_params

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this Instrument.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.embedded_inst.to_vector()

    def transform_inplace(self, s):
        """
        Update each Instrument element matrix `O` with `inv(s) * O * s`.

        Generally, the transform function updates the *parameters* of
        the operation such that the resulting operation matrix is altered as
        described above.  If such an update cannot be done (because
        the operation parameters do not allow for it), ValueError is raised.

        In this particular case any TP gauge transformation is possible,
        i.e. when `s` is an instance of `TPGaugeGroupElement` or
        corresponds to a TP-like transform matrix.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.

        Returns
        -------
        None
        """
        # I think we could do this but extracting the approprate parts of the
        # s and Sinv matrices... but haven't needed it yet.
        raise NotImplementedError("Cannot transform an EmbeddedInst yet...")

    def to_memoized_dict(self, mmg_memo):
        """Create a serializable dict with references to other objects in the memo.

        Parameters
        ----------
        mmg_memo: dict
            Memo dict from a ModelMemberGraph, i.e. keys are object ids and values
            are ModelMemberGraphNodes (which contain the serialize_id). This is NOT
            the same as other memos in ModelMember (e.g. copy, allocate_gpindices, etc.).

        Returns
        -------
        mm_dict: dict
            A dict representation of this ModelMember ready for serialization
            This must have at least the following fields:
            module, class, submembers, params, state_space, evotype
            Additional fields may be added by derived classes.
        """
        mm_dict = super().to_memoized_dict(mmg_memo)
        mm_dict['target_labels'] = self.target_labels
        mm_dict['member_labels'] = list(self.keys())  # labels of the submember effects

        return mm_dict

    def depolarize(self, amount):
        """
        Depolarize this Instrument by the given `amount`.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. All but the
            first element of each spam vector (often corresponding to the
            identity element) are multiplied by `amount` (if a float) or
            the corresponding `amount[i]` (if a tuple).

        Returns
        -------
        None
        """
        #Note: since each Mi is a linear function of MT and the Di, we can just
        # depolarize the MT and Di (self.param_ops) and re-init the elements.
        self.embedded_inst.depolarize(amount)

    def rotate(self, amount, mx_basis='gm'):
        """
        Rotate this instrument by the given `amount`.

        Parameters
        ----------
        amount : tuple of floats, optional
            Specifies the rotation "coefficients" along each of the non-identity
            Pauli-product axes.  The gate's matrix `G` is composed with a
            rotation operation `R`  (so `G` -> `dot(R, G)` ) where `R` is the
            unitary superoperator corresponding to the unitary operator
            `U = exp( sum_k( i * rotate[k] / 2.0 * Pauli_k ) )`.  Here `Pauli_k`
            ranges over all of the non-identity un-normalized Pauli operators.

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        Returns
        -------
        None
        """
        #Note: since each Mi is a linear function of MT and the Di, we can just
        # rotate the MT and Di (self.param_ops) and re-init the elements.
        self.embedded_inst.rotate(amount, mx_basis)
        
#Done I think??
    @classmethod  
    def _is_similar(self, other, rtol, atol):
        """ Returns True if `other` model member (which it guaranteed to be the same type as self) has
            the same local structure, i.e., not considering parameter values or submembers """
        return (self.target_labels == other.target_labels) and (self.state_space == other.state_space)

    def _oneline_contents(self):
        """ Summarizes the contents of this object in a single line.  Does not summarize submembers. """
        return "embeds %s into %s" % (str(self.target_labels), str(self.state_space))

    def __str__(self):
        """ Return string representation """
        s = "Embedded instrument with state space %s\n" % (self.state_space)
        s += " that embeds the following instrument into acting on the %s space\n" \
             % (str(self.target_labels))
        s += str(self.embedded_inst)
        return s
