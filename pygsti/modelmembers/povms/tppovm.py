"""
Defines the TPPOVM class
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    import torch as _torch
try:
    import torch as _torch
except ImportError:
    pass

import numpy as _np
from pygsti.modelmembers.torchable import Torchable as _Torchable
from pygsti.modelmembers.povms.basepovm import _BasePOVM
from pygsti.modelmembers.povms.fulleffect import FullPOVMEffect as _FullPOVMEffect
import warnings


class TPPOVM(_BasePOVM, _Torchable):
    """
    A POVM whose sum-of-effects is constrained to what, by definition, we call the "identity".

    Parameters
    ----------
    effects : dict of POVMEffects or array-like
        A dict (or list of key,value pairs) of the effect vectors.  The
        final effect vector will be stripped of any existing
        parameterization and turned into a ComplementPOVMEffect which has
        no additional parameters and is always equal to
        `identity - sum(other_effects`, where `identity` is the sum of
        `effects` when this __init__ call is made.

    evotype : Evotype or str, optional
        The evolution type.  If `None`, the evotype is inferred
        from the first effect vector.  If `len(effects) == 0` in this case,
        an error is raised.

    state_space : StateSpace, optional
        The state space for this POVM.  If `None`, the space is inferred
        from the first effect vector.  If `len(effects) == 0` in this case,
        an error is raised.

    Notes
    -----
    Just like TPState, we're restricted to the Pauli-product or Gell-Mann basis.
    
    We inherit from BasePOVM, which inherits from POVM, which inherits from OrderedDict.

    A TPPOVM "p" has an attribute p.complement_label that's set during construction.
    This label is such that e = p[p.complement_label] is a ComplementPOVMEffect, with
    an associated FullState object given in e.identity. If v = e.identity.to_vector(),
    then e's vector representation is
    
            v - sum(all non-complement effects in p).
    
    Under typical conditions v will be proportional to the first standard basis vector,
    and, in fact, if v is length "d," then we'll have v[0] == d ** 0.25. However, 
    neither of these conditions is strictly required by the API.
    """

    def __init__(self, effects, evotype=None, state_space=None, called_from_reduce=False):
        super(TPPOVM, self).__init__(effects, evotype, state_space, preserve_sum=True,
                                     called_from_reduce=called_from_reduce)

    def __reduce__(self):
        """ Needed for OrderedDict-derived classes (to set dict items) """
        assert(self.complement_label is not None)
        effects = [(lbl, effect.copy()) for lbl, effect in self.items()
                   if lbl != self.complement_label]

        #add complement effect as a std numpy array - it will get
        # re-created correctly by __init__ w/preserve_sum == True
        effects.append((self.complement_label,
                        self[self.complement_label].to_dense().reshape((-1, 1))))

        return (TPPOVM, (effects, self.evotype, self.state_space, True),
                {'_gpindices': self._gpindices, '_submember_rpindices': self._submember_rpindices})
    
    @property
    def dim(self):
        effect = next(iter(self.values()))
        return effect.dim
    
    def to_vector(self):
        effect_vecs = []
        for i, (lbl, effect) in enumerate(self.items()):
            if lbl != self.complement_label:
                assert isinstance(effect, _FullPOVMEffect)
                effect_vecs.append(effect.to_vector())
            else:
                assert i == len(self) - 1
        vec = _np.concatenate(effect_vecs)
        return vec

    def stateless_data(self) -> Tuple[int, _torch.Tensor, int]:
        num_effects = len(self)
        complement_effect = self[self.complement_label]
        identity = complement_effect.identity.to_vector()
        identity = identity.reshape((1, -1)) # make into a row vector
        t_identity = _torch.from_numpy(identity)
    
        dim = identity.size
        first_basis_vec = _np.zeros((1,dim))
        first_basis_vec[0,0] = dim ** 0.25
        TOL = 1e-15 * _np.sqrt(dim)
        if _np.linalg.norm(first_basis_vec - identity) > TOL:
            # Don't error out. The documentation for the class
            # clearly indicates that the meaning of "identity"
            # can be nonstandard.
            warnings.warn('Unexpected normalization!') 
        return (num_effects, t_identity, dim)

    @staticmethod
    def torch_base(sd: Tuple[int, _torch.Tensor, int], t_param: _torch.Tensor) -> _torch.Tensor:
        num_effects, t_identity, dim = sd
        t_param_mat = t_param.view(num_effects - 1, dim)
        t_func = t_identity - t_param_mat.sum(axis=0, keepdim=True)
        t = _torch.row_stack((t_param_mat, t_func))
        return t
