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

import numpy as _np
from pygsti.modelmembers.povms.basepovm import _BasePOVM
from pygsti.modelmembers.povms.effect import POVMEffect as _POVMEffect


class TPPOVM(_BasePOVM):
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
    
    @property
    def base(self):
        effectreps = [effect._rep for effect in self.values()]
        povm_mat = _np.row_stack([erep.state_rep.base for erep in effectreps])
        return povm_mat

    def torch_base(self, require_grad=False, torch_handle=None):
        if torch_handle is None:
            import torch as torch_handle
        if not require_grad:
            t = torch_handle.from_numpy(self.base)
            return t, []
        else:
            assert self.complement_label is not None
            complement_index = -1
            for i,k in enumerate(self.keys()):
                if k == self.complement_label:
                    complement_index = i
                    break
            assert complement_index >= 0

            num_effects = len(self)
            if complement_index != num_effects - 1:
                raise NotImplementedError()

            not_comp_selector = _np.ones(shape=(num_effects,), dtype=bool)
            not_comp_selector[complement_index] = False
            dim = self.dim
            first_basis_vec = torch_handle.zeros(size=(1, dim), dtype=torch_handle.double)
            first_basis_vec[0,0] = dim ** 0.25

            base = self.base
            t_param = torch_handle.from_numpy(base[not_comp_selector, :])
            t_param.requires_grad_(True)
            t_func = first_basis_vec - t_param.sum(axis=0, keepdim=True)
            t = torch_handle.row_stack((t_param, t_func))
            return t, [t_param]

