"""
GaugeGroup and derived objects, used primarily in gauge optimization
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

from pygsti.baseobjs import StateSpace as _StateSpace
from pygsti.modelmembers import operations as _op
from pygsti.baseobjs import statespace as _statespace
from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable
from pygsti.evotypes.evotype import Evotype as _Evotype


class GaugeGroup(_NicelySerializable):
    """
    A parameterized set (ideally a group) of gauge transformations.

    Specifies the "optimization space" explored by gauge optimization
    algorithms.  This base class is used to define the common interface of all
    types of gauge "groups" (even though they need not be groups in the
    mathematical sense).

    Parameters
    ----------
    name : str
        A name for this group - used for reporting what type of
        gauge optimization was performed.
    """

    def __init__(self, name):
        """
        Creates a new gauge group object

        Parameters
        ----------
        name : str
            A name for this group - used for reporting what type of
            gauge optimization was performed.
        """
        self.name = name

    @property
    def num_params(self):
        """
        Return the number of parameters (degrees of freedom) of this gauge group..

        Returns
        -------
        int
        """
        return 0

    def compute_element(self, param_vec):
        """
        Retrieve the element of this group corresponding to `param_vec`

        Parameters
        ----------
        param_vec : numpy.ndarray
            A 1D array of length :method:`num_params`.

        Returns
        -------
        GaugeGroupElement
        """
        return GaugeGroupElement()

    @property
    def initial_params(self):
        """
        Return a good (or standard) starting parameter vector, used to initialize a gauge optimization.

        Returns
        -------
        numpy.ndarray
            A 1D array of length :method:`num_params`.
        """
        return _np.array([], 'd')


class GaugeGroupElement(_NicelySerializable):
    """
    The element of a :class:`GaugeGroup`, which represents a single gauge transformation.
    """

    def __init__(self):
        """Creates a new GaugeGroupElement"""
        pass

    @property
    def transform_matrix(self):
        """
        The gauge-transform matrix.

        Returns
        -------
        numpy.ndarray
        """
        return None

    @property
    def transform_matrix_inverse(self):
        """
        The inverse of the gauge-transform matrix.

        Returns
        -------
        numpy.ndarray
        """
        return None

    def deriv_wrt_params(self, wrt_filter=None):
        """
        Computes the derivative of the gauge group at this element.

        That is, the derivative of a general element with respect to the gauge
        group's parameters, evaluated at this element.

        Parameters
        ----------
        wrt_filter : list or numpy.ndarray, optional
            Indices of the gauge group parameters to differentiate with respect to.
            If None, differentiation is performed with respect to all the group's parameters.

        Returns
        -------
        numpy.ndarray
        """
        return None

    def to_vector(self):
        """
        Get the parameter vector corresponding to this transform.

        Returns
        -------
        numpy.ndarray
        """
        return _np.array([], 'd')

    def from_vector(self, v):
        """
        Reinitialize this `GaugeGroupElement` using the the parameter vector `v`.

        Parameters
        ----------
        v : numpy.ndarray
            A 1D array of length :method:`num_params`

        Returns
        -------
        None
        """
        pass

    @property
    def num_params(self):
        """
        Return the number of parameters of this gauge group element.

        (This is equivalent to the number of parameters of the parent gauge group.)

        Returns
        -------
        int
        """
        return 0

    def inverse(self):
        """
        Creates a gauge group element that performs the inverse of this element.

        Returns
        -------
        InverseGaugeGroupElement
        """
        return InverseGaugeGroupElement(self)


class InverseGaugeGroupElement(GaugeGroupElement):
    """
    A gauge group element that represents the inverse action of another element.

    Parameters
    ----------
    gauge_group_el : GaugeGroupElement
        The element to invert.
    """

    def __init__(self, gauge_group_el):
        self.inverse_element = gauge_group_el

    @property
    def transform_matrix(self):
        """
        The gauge-transform matrix.

        Returns
        -------
        numpy.ndarray
        """
        return self.inverse_element.transform_matrix_inverse

    @property
    def transform_matrix_inverse(self):
        """
        The inverse of the gauge-transform matrix.

        Returns
        -------
        numpy.ndarray
        """
        return self.inverse_element.transform_matrix

    def deriv_wrt_params(self, wrt_filter=None):
        """
        Computes the derivative of the gauge group at this element.

        That is, the derivative of a general element with respect to the gauge
        group's parameters, evaluated at this element.

        Parameters
        ----------
        wrt_filter : list or numpy.ndarray, optional
            Indices of the gauge group parameters to differentiate with respect to.
            If None, differentiation is performed with respect to all the group's parameters.

        Returns
        -------
        numpy.ndarray
        """
        #Derivative of inv(M): d(inv_M) = inv_M * dM * inv_M
        Tinv = self.transform_matrix  # inverse of *original* transform
        dT = self.inverse_element.deriv_wrt_params(wrt_filter)  # shape (d*d, n)
        d, n = int(round(_np.sqrt(dT.shape[0]))), dT.shape[1]

        dT.shape = (d, d, n)  # call it (d1,d2,n)
        dT = _np.rollaxis(dT, 2)  # shape (n, d1, d2)
        deriv = -_np.dot(Tinv, _np.dot(dT, Tinv))  # d,d * (n,d,d * d,d) => d,d * n,d,d => d,n,d
        return _np.swapaxes(deriv, 1, 2).reshape(d * d, n)  # d,n,d => d,d,n => (d*d, n)

    def to_vector(self):
        """
        Get the parameter vector corresponding to this transform.

        Returns
        -------
        numpy.ndarray
        """
        return self.inverse_element.to_vector()

    def from_vector(self, v):
        """
        Reinitialize this `GaugeGroupElement` using the the parameter vector `v`.

        Parameters
        ----------
        v : numpy.ndarray
            A 1D array of length :method:`num_params`

        Returns
        -------
        None
        """
        return self.inverse_element.from_vector()

    @property
    def num_params(self):
        """
        Return the number of parameters of this gauge group element.

        (This is equivalent to the number of parameters of the parent gauge group.)

        Returns
        -------
        int
        """
        return self.inverse_element.num_params

    def inverse(self):
        """
        Creates a gauge group element that performs the inverse of this element.

        Returns
        -------
        GaugeGroupElement
        """
        return self.inverse_element  # inverting an inverse => back to original


class OpGaugeGroup(GaugeGroup):
    """
    A gauge group based on the parameterization of a single `LinearOperator`.

    The parameterization of this linear operator is used to parameterize the
    gauge-transform matrix.  This class is used as the base class for sevearl
    other of gauge group classes.

    Parameters
    ----------
    operation : LinearOperator
        The LinearOperator to base this Gauge group on.

    elementcls : class
        The element class to use when implementing the `element` method.

    name : str
        A name for this group - used for reporting what type of
        gauge optimization was performed.
    """

    def __init__(self, operation, elementcls, name):
        """
        Create a new `OpGaugeGroup`.

        Parameters
        ----------
        operation : LinearOperator
            The LinearOperator to base this Gauge group on.

        elementcls : class
            The element class to use when implementing the `compute_element` method.

        name : str
            A name for this group - used for reporting what type of
            gauge optimization was performed.
        """
        if not isinstance(operation, _op.LinearOperator):
            operation = _op.StaticArbitraryOp(operation, evotype='default', state_space=None)
        self._operation = operation
        self.element = elementcls
        GaugeGroup.__init__(self, name)

    @property
    def num_params(self):
        """
        Return the number of parameters (degrees of freedom) of this gauge group.

        Returns
        -------
        int
        """
        return self._operation.num_params

    def compute_element(self, param_vec):
        """
        Retrieve the element of this group corresponding to `param_vec`

        Parameters
        ----------
        param_vec : numpy.ndarray
            A 1D array of length :method:`num_params`.

        Returns
        -------
        GaugeGroupElement
        """
        elgate = self._operation.copy()
        elgate.from_vector(param_vec)
        return self.element(elgate)

    @property
    def initial_params(self):
        """
        Return a good (or standard) starting parameter vector, used to initialize a gauge optimization.

        Returns
        -------
        numpy.ndarray
            A 1D array of length :method:`num_params`.
        """
        return self._operation.to_vector()

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'state_space_dimension': int(self._operation.state_space.dim),
                      'evotype': str(self._operation.evotype)
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        #Note: this method assumes the (different) __init__ signature used by derived classes
        return cls(_statespace.default_space_for_dim(state['state_space_dimension']), state['evotype'])


class OpGaugeGroupElement(GaugeGroupElement):
    """
    The element type for `OpGaugeGroup`-derived gauge groups

    Parameters
    ----------
    operation : LinearOperator
        The operation to base this element on. It provides both parameterization
        information and the gauge transformation matrix itself.
    """

    def __init__(self, operation):
        """
        Create a new element based on `operation`

        Parameters
        ----------
        operation : LinearOperator
            The operation to base this element on. It provides both parameterization
            information and the gauge transformation matrix itself.
        """
        if not isinstance(operation, _op.LinearOperator):
            operation = _op.StaticArbitraryOp(operation, evotype='default', state_space=None)
        self._operation = operation
        self._inv_matrix = None
        GaugeGroupElement.__init__(self)

    @property
    def transform_matrix(self):
        """
        The gauge-transform matrix.

        Returns
        -------
        numpy.ndarray
        """
        return self._operation.to_dense(on_space='minimal')

    @property
    def transform_matrix_inverse(self):
        """
        The inverse of the gauge-transform matrix.

        Returns
        -------
        numpy.ndarray
        """
        if self._inv_matrix is None:
            self._inv_matrix = _np.linalg.inv(self._operation.to_dense(on_space='minimal'))
        return self._inv_matrix

    def deriv_wrt_params(self, wrt_filter=None):
        """
        Computes the derivative of the gauge group at this element.

        That is, the derivative of a general element with respect to the gauge
        group's parameters, evaluated at this element.

        Parameters
        ----------
        wrt_filter : list or numpy.ndarray, optional
            Indices of the gauge group parameters to differentiate with respect to.
            If None, differentiation is performed with respect to all the group's parameters.

        Returns
        -------
        numpy.ndarray
        """
        return self._operation.deriv_wrt_params(wrt_filter)

    def to_vector(self):
        """
        Get the parameter vector corresponding to this transform.

        Returns
        -------
        numpy.ndarray
        """
        return self._operation.to_vector()

    def from_vector(self, v):
        """
        Reinitialize this `GaugeGroupElement` using the the parameter vector `v`.

        Parameters
        ----------
        v : numpy.ndarray
            A 1D array of length :method:`num_params`

        Returns
        -------
        None
        """
        self._operation.from_vector(v)
        self._inv_matrix = None

    @property
    def num_params(self):
        """
        Return the number of parameters (degrees of freedom) of this element.

        Returns
        -------
        int
        """
        return self._operation.num_params

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'class': self.__class__.__name__,
                      'operation_matrix': self._encodemx(self._operation.to_dense())
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):  # memo holds already de-serialized objects
        operation_mx = cls._decodemx(state['operation_matrix'])
        return cls(operation_mx)


class FullGaugeGroup(OpGaugeGroup):
    """
    A fully-parameterized gauge group.

    Every element of the gauge transformation matrix is an independent parameter.

    Parameters
    ----------
    state_space : StateSpace
        The state space for this gauge group.  This is the state space that
        elements of the gauge group act on.  This should be the same as `mdl.state_space`
        where `mdl` is a :class:`Model` you want to gauge-transform.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.
    """

    def __init__(self, state_space, evotype='default'):
        state_space = _StateSpace.cast(state_space)
        operation = _op.FullArbitraryOp(_np.identity(state_space.dim, 'd'), None, evotype, state_space)
        OpGaugeGroup.__init__(self, operation, FullGaugeGroupElement, "Full")


class FullGaugeGroupElement(OpGaugeGroupElement):
    """
    Element of a :class:`FullGaugeGroup`

    Parameters
    ----------
    operation : LinearOperator
        The operation to base this element on. It provides both parameterization
        information and the gauge transformation matrix itself.
    """

    def __init__(self, operation):
        """
        Creates a new gauge group element based on `operation`, which
        is assumed to have the correct parameterization.
        """
        OpGaugeGroupElement.__init__(self, operation)


class TPGaugeGroup(OpGaugeGroup):
    """
    A gauge group spanning all trace-preserving (TP) gauge transformations.

    Implemented as a gauge transformation matrix whose first row is locked
    as `[1,0,0...0]` and where every other element is an independent parameter.

    Parameters
    ----------
    state_space : StateSpace
        The state space for this gauge group.  This is the state space that
        elements of the gauge group act on.  This should be the same as `mdl.state_space`
        where `mdl` is a :class:`Model` you want to gauge-transform.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.
    """

    def __init__(self, state_space, evotype='default'):
        state_space = _StateSpace.cast(state_space)
        operation = _op.FullTPOp(_np.identity(state_space.dim, 'd'), None, evotype, state_space)
        OpGaugeGroup.__init__(self, operation, TPGaugeGroupElement, "TP")


class TPGaugeGroupElement(OpGaugeGroupElement):
    """
    Element of a :class:`TPGaugeGroup`

    Parameters
    ----------
    operation : LinearOperator
        The operation to base this element on. It provides both parameterization
        information and the gauge transformation matrix itself.
    """

    def __init__(self, operation):
        """
        Creates a new gauge group element based on `operation`, which
        is assumed to have the correct parameterization.
        """
        OpGaugeGroupElement.__init__(self, operation)

    @property
    def transform_matrix_inverse(self):
        """
        The inverse of the gauge-transform matrix.

        Returns
        -------
        numpy.ndarray
        """
        if self._inv_matrix is None:
            self._inv_matrix = _np.linalg.inv(self._operation.to_dense())
            self._inv_matrix[0, :] = 0.0  # ensure invers is *exactly* TP
            self._inv_matrix[0, 0] = 1.0  # as otherwise small variations can get amplified
        return self._inv_matrix


class DiagGaugeGroup(OpGaugeGroup):
    """
    A gauge group consisting of just diagonal gauge-transform matrices.

    (Each diagonal element is a separate parameter.)

    Parameters
    ----------
    state_space : StateSpace
        The state space for this gauge group.  This is the state space that
        elements of the gauge group act on.  This should be the same as `mdl.state_space`
        where `mdl` is a :class:`Model` you want to gauge-transform.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.
    """

    def __init__(self, state_space, evotype='default'):
        state_space = _StateSpace.cast(state_space)
        dim = state_space.dim
        ltrans = _np.identity(dim, 'd')
        rtrans = _np.identity(dim, 'd')
        baseMx = _np.identity(dim, 'd')
        parameterArray = _np.zeros(dim, 'd')
        parameterToBaseIndicesMap = {i: [(i, i)] for i in range(dim)}
        operation = _op.LinearlyParamArbitraryOp(baseMx, parameterArray, parameterToBaseIndicesMap, ltrans, rtrans,
                                                 real=True, evotype=evotype, state_space=state_space)
        OpGaugeGroup.__init__(self, operation, DiagGaugeGroupElement, "Diagonal")


class DiagGaugeGroupElement(OpGaugeGroupElement):
    """
    Element of a :class:`DiagGaugeGroup`

    Parameters
    ----------
    operation : LinearOperator
        The operation to base this element on. It provides both parameterization
        information and the gauge transformation matrix itself.
    """

    def __init__(self, operation):
        """
        Creates a new gauge group element based on `operation`, which
        is assumed to have the correct parameterization.
        """
        OpGaugeGroupElement.__init__(self, operation)


class TPDiagGaugeGroup(TPGaugeGroup):
    """
    A gauge group consisting of just trace-preserving (TP) diagonal gauge-transform matrices.

    That is, where the first (`[0,0]`) element is fixed at 1.0,
    and each subsequent diagonal element is a separate parameter.

    Parameters
    ----------
    state_space : StateSpace
        The state space for this gauge group.  This is the state space that
        elements of the gauge group act on.  This should be the same as `mdl.state_space`
        where `mdl` is a :class:`Model` you want to gauge-transform.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.
    """

    def __init__(self, state_space, evotype='default'):
        """
        Create a new gauge group with gauge-transform dimension `dim`, which
        should be the same as `mdl.dim` where `mdl` is a :class:`Model` you
        might gauge-transform.
        """
        state_space = _StateSpace.cast(state_space)
        dim = state_space.dim
        ltrans = _np.identity(dim, 'd')
        rtrans = _np.identity(dim, 'd')
        baseMx = _np.identity(dim, 'd')
        parameterArray = _np.zeros(dim - 1, 'd')
        parameterToBaseIndicesMap = {i: [(i + 1, i + 1)] for i in range(dim - 1)}
        operation = _op.LinearlyParamArbitraryOp(baseMx, parameterArray, parameterToBaseIndicesMap, ltrans, rtrans,
                                                 real=True, evotype=evotype, state_space=state_space)
        OpGaugeGroup.__init__(self, operation, TPDiagGaugeGroupElement, "TP Diagonal")


class TPDiagGaugeGroupElement(TPGaugeGroupElement):
    """
    Element of a :class:`TPDiagGaugeGroup`

    Parameters
    ----------
    operation : LinearOperator
        The operation to base this element on. It provides both parameterization
        information and the gauge transformation matrix itself.
    """

    def __init__(self, operation):
        """
        Creates a new gauge group element based on `operation`, which
        is assumed to have the correct parameterization.
        """
        TPGaugeGroupElement.__init__(self, operation)


class UnitaryGaugeGroup(OpGaugeGroup):
    """
    A gauge group consisting of unitary gauge-transform matrices.

    This group includes those (superoperator) transformation matrices that
    correspond to unitary evolution.  Parameterization is performed via a
    Lindblad parametrizaton with only Hamiltonian terms.

    Parameters
    ----------
    state_space : StateSpace
        The state space for this gauge group.  This is the state space that
        elements of the gauge group act on.  This should be the same as `mdl.state_space`
        where `mdl` is a :class:`Model` you want to gauge-transform.

    basis : Basis or {"pp", "gm", "std"}
        The basis to use when parameterizing the Hamiltonian Lindblad terms.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.
    """

    def __init__(self, state_space, basis, evotype='default'):
        state_space = _StateSpace.cast(state_space)
        evotype = _Evotype.cast(str(evotype), default_prefer_dense_reps=True)  # since we use deriv_wrt_params
        errgen = _op.LindbladErrorgen.from_operation_matrix(
            _np.identity(state_space.dim, 'd'), "H", basis, mx_basis=basis, evotype=evotype)
        operation = _op.ExpErrorgenOp(errgen)
        OpGaugeGroup.__init__(self, operation, UnitaryGaugeGroupElement, "Unitary")

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'state_space_dimension': int(self._operation.state_space.dim),
                      'basis': self._operation.errorgen.coefficient_blocks[0]._basis.to_nice_serialization(),
                      'evotype': str(self._operation.evotype)
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        basis = _Basis.from_nice_serialization(state['basis'])
        return cls(_statespace.default_space_for_dim(state['state_space_dimension']), basis, state['evotype'])


class UnitaryGaugeGroupElement(OpGaugeGroupElement):
    """
    Element of a :class:`UnitaryGaugeGroup`

    Parameters
    ----------
    operation : LinearOperator
        The operation to base this element on. It provides both parameterization
        information and the gauge transformation matrix itself.
    """

    def __init__(self, operation):
        """
        Creates a new gauge group element based on `operation`, which
        is assumed to have the correct parameterization.
        """
        OpGaugeGroupElement.__init__(self, operation)


class SpamGaugeGroup(OpGaugeGroup):
    """
    Gauge transformations which scale the SPAM and non-unital portions of the gates in a gate set.

    A 2-dimensional gauge group spanning transform matrices of the form:
    [ [ a 0 ... 0]   where a and b are the 2 parameters.  These diagonal
      [ 0 b ... 0]   transform matrices do not affect the SPAM operations
      [ . . ... .]   much more than typical near-unital and TP operations, and
      [ 0 0 ... b] ] so we call this group of transformations the "SPAM gauge".

    Parameters
    ----------
    state_space : StateSpace
        The state space for this gauge group.  This is the state space that
        elements of the gauge group act on.  This should be the same as `mdl.state_space`
        where `mdl` is a :class:`Model` you want to gauge-transform.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.
    """

    def __init__(self, state_space, evotype='default'):
        """
        Create a new gauge group with gauge-transform dimension `dim`, which
        should be the same as `mdl.dim` where `mdl` is a :class:`Model` you
        might gauge-transform.
        """
        state_space = _StateSpace.cast(state_space)
        dim = state_space.dim
        ltrans = _np.identity(dim, 'd')
        rtrans = _np.identity(dim, 'd')
        baseMx = _np.identity(dim, 'd')
        parameterArray = _np.zeros(2, 'd')
        parameterToBaseIndicesMap = {0: [(0, 0)],
                                     1: [(i, i) for i in range(1, dim)]}
        operation = _op.LinearlyParamArbitraryOp(baseMx, parameterArray, parameterToBaseIndicesMap, ltrans, rtrans,
                                                 real=True, evotype=evotype, state_space=state_space)
        OpGaugeGroup.__init__(self, operation, SpamGaugeGroupElement, "Spam")


class SpamGaugeGroupElement(OpGaugeGroupElement):
    """
    Element of a :class:`SpamGaugeGroup`

    Parameters
    ----------
    operation : LinearOperator
        The operation to base this element on. It provides both parameterization
        information and the gauge transformation matrix itself.
    """

    def __init__(self, operation):
        """
        Creates a new gauge group element based on `operation`, which
        is assumed to have the correct parameterization.
        """
        OpGaugeGroupElement.__init__(self, operation)


class TPSpamGaugeGroup(OpGaugeGroup):
    """
    Similar to :class:`SpamGaugeGroup` except with TP constrains.

    This means the `[0,0]` element of each transform matrix is fixed at 1.0
    (so all gauge transforms are trace preserving), leaving just a single degree
    of freedom.

    Parameters
    ----------
    state_space : StateSpace
        The state space for this gauge group.  This is the state space that
        elements of the gauge group act on.  This should be the same as `mdl.state_space`
        where `mdl` is a :class:`Model` you want to gauge-transform.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.
    """

    def __init__(self, state_space, evotype='default'):
        """
        Create a new gauge group with gauge-transform dimension `dim`, which
        should be the same as `mdl.dim` where `mdl` is a :class:`Model` you
        might gauge-transform.
        """
        state_space = _StateSpace.cast(state_space)
        dim = state_space.dim
        ltrans = _np.identity(dim, 'd')
        rtrans = _np.identity(dim, 'd')
        baseMx = _np.identity(dim, 'd')
        parameterArray = _np.zeros(1, 'd')
        parameterToBaseIndicesMap = {0: [(i, i) for i in range(1, dim)]}
        operation = _op.LinearlyParamArbitraryOp(baseMx, parameterArray, parameterToBaseIndicesMap, ltrans, rtrans,
                                                 real=True, evotype=evotype, state_space=state_space)
        OpGaugeGroup.__init__(self, operation, TPSpamGaugeGroupElement, "TP Spam")


class TPSpamGaugeGroupElement(OpGaugeGroupElement):
    """
    Element of :class:`TPSpamGaugeGroup`

    Parameters
    ----------
    operation : LinearOperator
        The operation to base this element on. It provides both parameterization
        information and the gauge transformation matrix itself.
    """

    def __init__(self, operation):
        """
        Creates a new gauge group element based on `operation`, which
        is assumed to have the correct parameterization.
        """
        OpGaugeGroupElement.__init__(self, operation)


class TrivialGaugeGroup(GaugeGroup):
    """
    A trivial gauge group with no degrees of freedom.

    Useful for telling pyGSTi that you don't want to do any gauge optimization
    within the framework common to the other gauge groups. Using a
    `TrivialGaugeGroup` instead of `None` in gauge optimization will prevent
    pyGSTi from wondering if you meant to not-gauge-optimize and displaying
    warning messages.

    Parameters
    ----------
    state_space : StateSpace
        The state space for this gauge group.  This is the state space that
        elements of the gauge group act on.  This should be the same as `mdl.state_space`
        where `mdl` is a :class:`Model` you want to gauge-transform.
    """

    def __init__(self, state_space):
        state_space = _StateSpace.cast(state_space)
        self.state_space = state_space
        GaugeGroup.__init__(self, "Trivial")

    @property
    def num_params(self):
        """
        Return the number of parameters (degrees of freedom) of this gauge group.

        Returns
        -------
        int
        """
        return 0

    def compute_element(self, param_vec):
        """
        Retrieve the element of this group corresponding to `param_vec`

        Parameters
        ----------
        param_vec : numpy.ndarray
            A 1D array of length :method:`num_params`.

        Returns
        -------
        TrivialGaugeGroupElement
        """
        assert(len(param_vec) == 0)
        return TrivialGaugeGroupElement(self.state_space.dim)

    @property
    def initial_params(self):
        """
        Return a good (or standard) starting parameter vector, used to initialize a gauge optimization.

        Returns
        -------
        numpy.ndarray
            A 1D array of length :method:`num_params`.
        """
        return _np.empty(0, 'd')

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'state_space': self.state_space.to_nice_serialization()})
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        state_space = _statespace.StateSpace.from_nice_serialization(state['state_space'])
        return cls(state_space)


class TrivialGaugeGroupElement(GaugeGroupElement):
    """
    Element of :class:`TrivialGaugeGroup`

    Parameters
    ----------
    dim : int
        The Hilbert-Schmidt space dimension of the gauge group.
    """

    def __init__(self, dim):
        """
        Creates a new trivial gauge group element of dimension `dim`.
        (so transform matirx is a `dim` by `dim` identity matrix).
        """
        self._matrix = _np.identity(dim, 'd')
        GaugeGroupElement.__init__(self)

    @property
    def transform_matrix(self):
        """
        The gauge-transform matrix.

        Returns
        -------
        numpy.ndarray
        """
        return self._matrix

    @property
    def transform_matrix_inverse(self):
        """
        The inverse of the gauge-transform matrix.

        Returns
        -------
        numpy.ndarray
        """
        return self._matrix  # inverse of identity is itself!

    def deriv_wrt_params(self, wrt_filter=None):
        """
        Computes the derivative of the gauge group at this element.

        That is, the derivative of a general element with respect to the gauge
        group's parameters, evaluated at this element.

        Parameters
        ----------
        wrt_filter : list or numpy.ndarray, optional
            Indices of the gauge group parameters to differentiate with respect to.
            If None, differentiation is performed with respect to all the group's parameters.

        Returns
        -------
        numpy.ndarray
        """
        return _np.empty(0, 'd')

    def to_vector(self):
        """
        Get the parameter vector corresponding to this transform.

        Returns
        -------
        numpy.ndarray
        """
        return _np.empty(0, 'd')

    def from_vector(self, v):
        """
        Reinitialize this `GaugeGroupElement` using the the parameter vector `v`.

        Parameters
        ----------
        v : numpy.ndarray
            A 1D array of length :method:`num_params`

        Returns
        -------
        None
        """
        assert(len(v) == 0)

    @property
    def num_params(self):
        """
        Return the number of parameters (degrees of freedom) of this element.

        Returns
        -------
        int
        """
        return 0

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'operation_dimension': self._matrix.shape[0]})
        return state

    @classmethod
    def _from_nice_serialization(cls, state):  # memo holds already de-serialized objects
        return cls(state['operation_dimension'])
