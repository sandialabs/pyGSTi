from pygsti.modelmembers.modelmember import ModelMember
from typing import TypeVar, Tuple

try:
    import torch
    torch_handle = torch
    Tensor = torch.Tensor
except ImportError:
    torch_handle = None
    Tensor = TypeVar('Tensor')  # we'll access this for type annotations elsewhere.


class Torchable(ModelMember):

    Tensor = Tensor
    torch_handle = torch

    def stateless_data(self) -> Tuple:
        """
        Return the data of this model that is considered considered constant for purposes
        of model fitting.

        Note: the word "stateless" here is used in the sense of object-oriented programming.
        """
        raise NotImplementedError()   

    @staticmethod
    def torch_base(sd : Tuple, t_param : Tensor) -> Tensor:
        """
        Suppose "obj" is an instance of some ModelMember subclass. If we compute

            sd = obj.stateless_data()
            vec = obj.to_vector()
            t_param = torch.from_numpy(vec)
            T = type(obj).torch_base(sd, t_param, grad)

        then T will be a PyTorch Tensor that represents "obj" in a canonical numerical way.

        The meaning of "canonical" is implementation dependent. If type(obj) implements
        the ``.base`` attribute, then a reasonable implementation will probably satisfy

            np.allclose(obj.base, T.numpy()).
        """
        raise NotImplementedError()
