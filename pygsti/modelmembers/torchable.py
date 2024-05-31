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
from typing import Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    import torch as _torch

from pygsti.modelmembers.modelmember import ModelMember


class Torchable(ModelMember):

    def stateless_data(self) -> Tuple:
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
            sd = obj.stateless_data()
            t = type(obj).torch_base(sd, t_param)

        then t will be a PyTorch Tensor that represents "obj" in a canonical numerical way.

        The meaning of "canonical" is implementation dependent. If type(obj) implements
        the ``.base`` attribute, then a reasonable implementation will probably satisfy

            np.allclose(obj.base, t.numpy()).
        """
        raise NotImplementedError()
