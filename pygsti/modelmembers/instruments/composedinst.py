"""
Defines the ComposedInst class
"""

import collections as _collections
import numpy as _np

from pygsti.modelmembers import modelmember as _mm
from pygsti.modelmembers import operations as _op
from pygsti.modelmembers import states as _state
from pygsti.evotypes import Evotype as _Evotype
from pygsti.baseobjs import statespace as _statespace
from pygsti.tools import matrixtools as _mt
from pygsti.tools import slicetools as _slct
from pygsti.baseobjs.label import Label as _Label
from pygsti.baseobjs.statespace import StateSpace as _StateSpace

class ComposedInst(_mm.ModelMember, _collections.OrderedDict):
    """
    A new class for representing a quantum instrument, the mathematical description 
    of an intermediate measurement. This class relies on the "auxiliary picture" where 
    a n-qubit m-outcome intermediate measurement corresponds to a circuit diagram with
    n plus m auxiliary qubits. 

    Using this picture allows us to extract error generators corresponding to the circuit
    diagram and lays the ground work for a CPTPLND parameterization. 

    Parameters
    ----------
    member_ops : dict of LinearOperator objects
        A dict (or list of key,value pairs) of the gates.
    inst_type : str
        A string, must be either 'Ipc' (two qubit parity check) 
        or 'Iz' (Z-basis one qubit measurement)
    evotype : Evotype or str, optional
        The evolution type.  If `None`, the evotype is inferred
        from the first instrument member.  If `len(member_ops) == 0` in this case,
        an error is raised.

    state_space : StateSpace, optional
        The state space for this POVM.  If `None`, the space is inferred
        from the first instrument member.  If `len(member_ops) == 0` in this case,
        an error is raised.

    items : list or dict, optional
        Initial values.  This should only be used internally in de-serialization.
    """
 
def __init__(self, evotype=None, state_space=None, items=None): 
    _collections.OrderedDict.__init__(self, items)
    _mm.ModelMember.__init__(self, state_space, evotype)
    self.from_vector(_np.ndarray.flatten(_np.identity(4)))

def to_vector(self): 
        """
        Gives the underlying vector of parameters. 

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        raise NotImplementedError("Not yet implemented!")

def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize the Instrument using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params().

        close : bool, optional
            Whether `v` is close to this Instrument's current
            set of parameters.  Under some circumstances, when this
            is true this call can be completed more quickly.

        dirty_value : bool, optional
            The value to set this object's "dirty flag" to before exiting this
            call.  This is passed as an argument so it can be updated *recursively*.
            Leave this set to `True` unless you know what you're doing.

        Returns
        -------
        None
        """
        self._ptr[:,:] = v.reshape(4,4)
        self.dirty = dirty_value


#functions: transform inplace (gauge optimization), str (print out), submembers (returns instrument elements) 