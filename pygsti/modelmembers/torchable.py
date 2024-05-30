from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    import torch as _torch

from pygsti.modelmembers.modelmember import ModelMember


class Torchable(ModelMember):

    # Try to import torch. If we succeed, save a handle to it for later use. If we fail, then 
    # set a flag indicating as much so we don't have to write try-except statements for torch
    # imports in other files.
    try:
        import torch
        torch_handle = torch
    except ImportError:
        torch_handle = None


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
