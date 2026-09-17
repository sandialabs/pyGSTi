"""
Defines the interface that ModelMembers must satisfy to be compatible with
the PyTorch-backed forward simulator in pyGSTi/forwardsims/torchfwdsim.py.
"""
#***************************************************************************************************
# Copyright 2024, National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


from __future__ import annotations
from typing import Tuple, TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    import torch as _torch
try:
    import torch as _torch
except ImportError:
    pass

import numpy as _np
from pygsti.modelmembers.modelmember import ModelMember
from pygsti.tools import slicetools as _slct


class Torchable(ModelMember):

    def stateless_data(self, real_dtype: _torch.dtype, device: _torch.Device) -> Tuple[Any, ...]:
        """
        Return this ModelMember's data that is considered constant for purposes of model fitting.

        Note: the word "stateless" here is used in the sense of object-oriented programming.
        """
        raise NotImplementedError()

    @staticmethod
    def torch_base(sd : Tuple, t_param : _torch.Tensor) -> _torch.Tensor:
        """
        Suppose "obj" is an instance of some Torchable subclass. If we compute

            vec = obj.to_vector()
            t_param = torch.from_numpy(vec)
            sd = obj.stateless_data(torch.float64, 'cpu')
            t = type(obj).torch_base(sd, t_param)

        then t will be a PyTorch Tensor that represents "obj" in a canonical numerical way.

        The meaning of "canonical" is implementation dependent. If type(obj) implements
        the ``.base`` attribute, then a reasonable implementation will probably satisfy

            np.allclose(obj.base, t.numpy()).
        """
        raise NotImplementedError()


class StaticTorchable(Torchable):
    """
    Mixin for static (non-parameterized) operations on the dense torch path.

    A static op contributes a constant dense superoperator and has no free parameters, so its
    `torch_base` simply returns the baked-in matrix and ignores the (empty) parameter slice.
    """

    def stateless_data(self, real_dtype: _torch.dtype, device: _torch.Device) -> Tuple[_torch.Tensor]:
        mx = _np.ascontiguousarray(self.to_dense('HilbertSchmidt'))
        if _np.iscomplexobj(mx):
            dtype = _torch.complex64 if real_dtype.itemsize == 4 else _torch.complex128
        else:
            dtype = real_dtype
        return (_torch.from_numpy(mx).to(dtype=dtype, device=device),)

    @staticmethod
    def torch_base(sd : Tuple, t_param : _torch.Tensor) -> _torch.Tensor:
        return sd[0]


class StackedMemberDictTorchable(Torchable):
    """
    Mixin for Torchable classes that are themselves ordered dicts of Torchable submembers sharing
    one parameter vector, keyed by ``self._submember_rpindices`` (e.g. Instrument, TPInstrument).

    ``torch_base`` stacks the members' individual torch_base tensors in ``self.keys()`` order; users
    of the stacked tensor (e.g. TorchForwardSimulator) rely on that order to recover per-member results.
    """

    def stateless_data(self, real_dtype: _torch.dtype, device: _torch.Device) -> Tuple[Any, ...]:
        member_data = []
        for (_, member), inds in zip(self.items(), self._submember_rpindices):
            assert isinstance(member, Torchable), \
                "Every %s member must be Torchable to use the Torch forward simulator; got %s" \
                % (type(self).__name__, type(member).__name__)
            idx = _torch.as_tensor(_slct.to_array(inds), dtype=_torch.long, device=device)
            member_data.append((type(member), member.stateless_data(real_dtype, device), idx))
        return (tuple(member_data),)

    @staticmethod
    def torch_base(sd: Tuple[Any, ...], t_param: _torch.Tensor) -> _torch.Tensor:
        (member_data,) = sd
        mats = [mtype.torch_base(msd, t_param[idx]) for (mtype, msd, idx) in member_data]
        return _torch.stack(mats)
