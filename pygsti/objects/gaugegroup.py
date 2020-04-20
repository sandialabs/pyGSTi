""" GaugeGroup and derived objects, used primarily in gauge optimization """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
from . import operation as _op


class GaugeGroup(object):
    """
    A GaugeGroup describes a parameterized set (ideally a group) of gauge
    transformations which specifies the "optimization space" explored by
    gauge optimization algorithms.  This base class is used to define the
    common interface of all types of gauge "groups" (even though they need
    not be groups in the mathematical sense).
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

    def num_params(self):
        """
        Return the number of parameters (degrees of freedom) of this
        `GaugeGroup`

        Returns
        -------
        int
        """
        return 0

    def get_element(self, param_vec):
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

    def get_initial_params(self):
        """
        Return a good (or standard) starting parameter vector, used for
        starting a gauge optimization.

        Returns
        -------
        numpy.ndarray
            A 1D array of length :method:`num_params`.
        """
        return _np.array([], 'd')


class GaugeGroupElement(object):
    """
    The element of a :class:`GaugeGroup`, which represents a single gauge
    transformation.
    """

    def __init__(self):
        """Creates a new GaugeGroupElement"""
        pass

    def get_transform_matrix(self):
        """Return the gauge-transform matrix"""
        return None

    def get_transform_matrix_inverse(self):
        """Return the inverse of the gauge-transform matrix"""
        return None

    def deriv_wrt_params(self, wrt_filter=None):
        """
        Return the derivative of the group of gauge transformations (with
        respect to the group's parameters) at this element.
        """
        return None

    def to_vector(self):
        """Return the parameter vector corresponding to this transform."""
        return _np.array([], 'd')

    def from_vector(self, v):
        """
        Reinitialize this `GaugeGroupElement` using the the parameter
        vector `v`.

        Parameters
        ----------
        v : numpy.ndarray
            A 1D array of length :method:`num_params`
        """
        pass

    def num_params(self):
        """
        Return the number of parameters of this gauge group element (equivalent
        to the number of parameters of it's gauge group).

        Returns
        -------
        int
        """
        return 0


class OpGaugeGroup(GaugeGroup):
    """
    A gauge group based on the parameterization of a single `LinearOperator`, which is
    used to parameterize the gauge-transform matrix.  This class is used as
    the base class for sevearl other of gauge group classes.
    """

    def __init__(self, gate, elementcls, name):
        """
        Create a new `OpGaugeGroup`.

        Parameters
        ----------
        gate : LinearOperator
            The LinearOperator to base this Gauge group on.

        elementcls : class
            The element class to use when implementing the `get_element` method.

        name : str
            A name for this group - used for reporting what type of
            gauge optimization was performed.
        """
        if not isinstance(gate, _op.LinearOperator):
            gate = _op.StaticDenseOp(gate)
        self.gate = gate
        self.element = elementcls
        GaugeGroup.__init__(self, name)

    def num_params(self):
        """ See :method:`GaugeGroup.num_params` """
        return self.gate.num_params()

    def get_element(self, param_vec):
        """ See :method:`GaugeGroup.get_element` """
        elgate = self.gate.copy()
        elgate.from_vector(param_vec)
        return self.element(elgate)

    def get_initial_params(self):
        """ See :method:`GaugeGroup.get_initial_params` """
        return self.gate.to_vector()


class OpGaugeGroupElement(GaugeGroupElement):
    """ The element type for `OpGaugeGroup`-derived gauge groups """

    def __init__(self, gate):
        """
        Create a new element based on `gate`

        Parameters
        ----------
        gate : LinearOperator
            The gate to base this element on. It provides both parameterization
            information and the gauge transformation matrix itself.
        """
        if not isinstance(gate, _op.LinearOperator):
            gate = _op.StaticDenseOp(gate)
        self.gate = gate
        self._inv_matrix = None
        GaugeGroupElement.__init__(self)

    def get_transform_matrix(self):
        """ See :method:`GaugeGroupElement.get_transform_matrix` """
        return self.gate.todense()

    def get_transform_matrix_inverse(self):
        """ See :method:`GaugeGroupElement.get_transform_matrix_inverse` """
        if self._inv_matrix is None:
            self._inv_matrix = _np.linalg.inv(self.gate.todense())
        return self._inv_matrix

    def deriv_wrt_params(self, wrt_filter=None):
        """ See :method:`GaugeGroupElement.deriv_wrt_params` """
        return self.gate.deriv_wrt_params(wrt_filter)

    def to_vector(self):
        """ See :method:`GaugeGroupElement.to_vector` """
        return self.gate.to_vector()

    def from_vector(self, v):
        """ See :method:`GaugeGroupElement.from_vector` """
        self.gate.from_vector(v)
        self._inv_matrix = None

    def num_params(self):
        """ See :method:`GaugeGroupElement.num_params` """
        return self.gate.num_params()


class FullGaugeGroup(OpGaugeGroup):
    """
    A fully-parameterized gauge group, where every element of the gauge
    transformation matrix is an independent parameter.
    """

    def __init__(self, dim):
        """
        Create a new gauge group with gauge-transform dimension `dim`, which
        should be the same as `mdl.dim` where `mdl` is a :class:`Model` you
        might gauge-transform.
        """
        from . import operation as _op  # b/c operation.py imports gaugegroup
        gate = _op.FullDenseOp(_np.identity(dim, 'd'))
        OpGaugeGroup.__init__(self, gate, FullGaugeGroupElement, "Full")


class FullGaugeGroupElement(OpGaugeGroupElement):
    """ Element of a :class:`FullGaugeGroup` """

    def __init__(self, gate):
        """
        Creates a new gauge group element based on `gate`, which
        is assumed to have the correct parameterization.
        """
        OpGaugeGroupElement.__init__(self, gate)


class TPGaugeGroup(OpGaugeGroup):
    """
    A gauge group spanning all trace-preserving (TP) gauge transformation,
    implemented as a gauge transformation matrix whose first row is locked
    as `[1,0,0...0]` and where every other element is an independent parameter.
    """

    def __init__(self, dim):
        """
        Create a new gauge group with gauge-transform dimension `dim`, which
        should be the same as `mdl.dim` where `mdl` is a :class:`Model` you
        might gauge-transform.
        """
        from . import operation as _op  # b/c operation.py imports gaugegroup
        gate = _op.TPDenseOp(_np.identity(dim, 'd'))
        OpGaugeGroup.__init__(self, gate, TPGaugeGroupElement, "TP")


class TPGaugeGroupElement(OpGaugeGroupElement):
    """ Element of a :class:`TPGaugeGroup` """

    def __init__(self, gate):
        """
        Creates a new gauge group element based on `gate`, which
        is assumed to have the correct parameterization.
        """
        OpGaugeGroupElement.__init__(self, gate)

    def get_transform_matrix_inverse(self):
        """ See :method:`GaugeGroupElement.get_transform_matrix_inverse` """
        if self._inv_matrix is None:
            self._inv_matrix = _np.linalg.inv(self.gate.todense())
            self._inv_matrix[0, :] = 0.0  # ensure invers is *exactly* TP
            self._inv_matrix[0, 0] = 1.0  # as otherwise small variations can get amplified
        return self._inv_matrix


class DiagGaugeGroup(OpGaugeGroup):
    """
    A gauge group consisting of just diagonal gauge-transform matrices, where
    each diagonal element is a separate parameter.
    """

    def __init__(self, dim):
        """
        Create a new gauge group with gauge-transform dimension `dim`, which
        should be the same as `mdl.dim` where `mdl` is a :class:`Model` you
        might gauge-transform.
        """
        from . import operation as _op  # b/c operation.py imports gaugegroup
        ltrans = _np.identity(dim, 'd')
        rtrans = _np.identity(dim, 'd')
        baseMx = _np.identity(dim, 'd')
        parameterArray = _np.zeros(dim, 'd')
        parameterToBaseIndicesMap = {i: [(i, i)] for i in range(dim)}
        gate = _op.LinearlyParamDenseOp(baseMx, parameterArray,
                                        parameterToBaseIndicesMap,
                                        ltrans, rtrans, real=True)
        OpGaugeGroup.__init__(self, gate, DiagGaugeGroupElement, "Diagonal")


class DiagGaugeGroupElement(OpGaugeGroupElement):
    """ Element of a :class:`DiagGaugeGroup` """

    def __init__(self, gate):
        """
        Creates a new gauge group element based on `gate`, which
        is assumed to have the correct parameterization.
        """
        OpGaugeGroupElement.__init__(self, gate)


class TPDiagGaugeGroup(TPGaugeGroup):
    """
    A gauge group consisting of just trace-preserving (TP) diagonal
    gauge-transform matrices, where the first (`[0,0]`) element is fixed at 1.0,
    and each subsequent diagonal element is a separate parameter.
    """

    def __init__(self, dim):
        """
        Create a new gauge group with gauge-transform dimension `dim`, which
        should be the same as `mdl.dim` where `mdl` is a :class:`Model` you
        might gauge-transform.
        """
        from . import operation as _op  # b/c operation.py imports gaugegroup
        ltrans = _np.identity(dim, 'd')
        rtrans = _np.identity(dim, 'd')
        baseMx = _np.identity(dim, 'd')
        parameterArray = _np.zeros(dim - 1, 'd')
        parameterToBaseIndicesMap = {i: [(i + 1, i + 1)] for i in range(dim - 1)}
        gate = _op.LinearlyParamDenseOp(baseMx, parameterArray,
                                        parameterToBaseIndicesMap,
                                        ltrans, rtrans, real=True)
        OpGaugeGroup.__init__(self, gate, TPDiagGaugeGroupElement, "TP Diagonal")


class TPDiagGaugeGroupElement(TPGaugeGroupElement):
    """ Element of a :class:`TPDiagGaugeGroup` """

    def __init__(self, gate):
        """
        Creates a new gauge group element based on `gate`, which
        is assumed to have the correct parameterization.
        """
        TPGaugeGroupElement.__init__(self, gate)


class UnitaryGaugeGroup(OpGaugeGroup):
    """
    A gauge group consisting of unitary gauge-transform matrices - that is,
    those superoperator transformation matrices which correspond to
    unitary evolution.  Parameterization is performed via a Lindblad
    parametrizaton with only Hamiltonian terms.
    """

    def __init__(self, dim, basis):
        """
        Create a new gauge group with gauge-transform dimension `dim`, which
        should be the same as `mdl.dim` where `mdl` is a :class:`Model` you
        might gauge-transform.
        """
        from . import operation as _op  # b/c operation.py imports gaugegroup
        gate = _op.LindbladDenseOp.from_operation_matrix(
            None, _np.identity(dim, 'd'), ham_basis=basis, nonham_basis=None,
            param_mode="cptp", mx_basis=basis)
        OpGaugeGroup.__init__(self, gate, UnitaryGaugeGroupElement, "Unitary")


class UnitaryGaugeGroupElement(OpGaugeGroupElement):
    """ Element of a :class:`UnitaryGaugeGroup` """

    def __init__(self, gate):
        """
        Creates a new gauge group element based on `gate`, which
        is assumed to have the correct parameterization.
        """
        OpGaugeGroupElement.__init__(self, gate)


class SpamGaugeGroup(OpGaugeGroup):
    """
    A 2-dimensional gauge group spanning transform matrices of the form:
    [ [ a 0 ... 0]   where a and b are the 2 parameters.  These diagonal
      [ 0 b ... 0]   transform matrices do not affect the SPAM operations
      [ . . ... .]   much more than typical near-unital and TP gates, and
      [ 0 0 ... b] ] so we call this group of transformations the "SPAM gauge".
    """

    def __init__(self, dim):
        """
        Create a new gauge group with gauge-transform dimension `dim`, which
        should be the same as `mdl.dim` where `mdl` is a :class:`Model` you
        might gauge-transform.
        """
        from . import operation as _op  # b/c operation.py imports gaugegroup
        ltrans = _np.identity(dim, 'd')
        rtrans = _np.identity(dim, 'd')
        baseMx = _np.identity(dim, 'd')
        parameterArray = _np.zeros(2, 'd')
        parameterToBaseIndicesMap = {0: [(0, 0)],
                                     1: [(i, i) for i in range(1, dim)]}
        gate = _op.LinearlyParamDenseOp(baseMx, parameterArray,
                                        parameterToBaseIndicesMap,
                                        ltrans, rtrans, real=True)
        OpGaugeGroup.__init__(self, gate, SpamGaugeGroupElement, "Spam")


class SpamGaugeGroupElement(OpGaugeGroupElement):
    """ Element of a :class:`SpamGaugeGroup` """

    def __init__(self, gate):
        """
        Creates a new gauge group element based on `gate`, which
        is assumed to have the correct parameterization.
        """
        OpGaugeGroupElement.__init__(self, gate)


class TPSpamGaugeGroup(OpGaugeGroup):
    """
    A gauge group similar to :class:`SpamGaugeGroup` except the `[0,0]` element
    of each transform matrix is fixed at 1.0 (so all gauge transforms are trace
    preserving), leaving just a single degree of freedom.
    """

    def __init__(self, dim):
        """
        Create a new gauge group with gauge-transform dimension `dim`, which
        should be the same as `mdl.dim` where `mdl` is a :class:`Model` you
        might gauge-transform.
        """
        from . import operation as _op  # b/c operation.py imports gaugegroup
        ltrans = _np.identity(dim, 'd')
        rtrans = _np.identity(dim, 'd')
        baseMx = _np.identity(dim, 'd')
        parameterArray = _np.zeros(1, 'd')
        parameterToBaseIndicesMap = {0: [(i, i) for i in range(1, dim)]}
        gate = _op.LinearlyParamDenseOp(baseMx, parameterArray,
                                        parameterToBaseIndicesMap,
                                        ltrans, rtrans, real=True)
        OpGaugeGroup.__init__(self, gate, TPSpamGaugeGroupElement, "TP Spam")


class TPSpamGaugeGroupElement(OpGaugeGroupElement):
    """ Element of :class:`TPSpamGaugeGroup` """

    def __init__(self, gate):
        """
        Creates a new gauge group element based on `gate`, which
        is assumed to have the correct parameterization.
        """
        OpGaugeGroupElement.__init__(self, gate)


class TrivialGaugeGroup(GaugeGroup):
    """
    A trivial gauge group with no degrees of freedom.  Useful
    for telling pyGSTi that you don't want to do any gauge optimization
    within the framework common to the other gauge groups. Using a
    `TrivialGaugeGroup` instead of `None` in gauge optimization will
    prevent pyGSTi from wondering if you meant to not-gauge-optimize and
    displaying warning messages.
    """

    def __init__(self, dim):
        """
        Create a new gauge group with gauge-transform dimension `dim`, which
        should be the same as `mdl.dim` where `mdl` is a :class:`Model` you
        might gauge-transform.
        """
        self.dim = dim
        GaugeGroup.__init__(self, "Trivial")

    def num_params(self):
        """ See :method:`GaugeGroup.num_params` """
        return 0

    def get_element(self, param_vec):
        """ See :method:`GaugeGroup.get_element` """
        assert(len(param_vec) == 0)
        return TrivialGaugeGroupElement(self.dim)

    def get_initial_params(self):
        """ See :method:`GaugeGroup.get_initial_params` """
        return _np.empty(0, 'd')


class TrivialGaugeGroupElement(GaugeGroupElement):
    """ Element of :class:`TrivialGaugeGroup` """

    def __init__(self, dim):
        """
        Creates a new trivial gauge group element of dimension `dim`.
        (so transform matirx is a `dim` by `dim` identity matrix).
        """
        self._matrix = _np.identity(dim, 'd')
        GaugeGroupElement.__init__(self)

    def get_transform_matrix(self):
        """ See :method:`GaugeGroupElement.get_transform_matrix` """
        return self._matrix

    def get_transform_matrix_inverse(self):
        """ See :method:`GaugeGroupElement.get_transform_matrix_inverse` """
        return self._matrix  # inverse of identity is itself!

    def deriv_wrt_params(self, wrt_filter=None):
        """ See :method:`GaugeGroupElement.deriv_wrt_params` """
        return _np.empty(0, 'd')

    def to_vector(self):
        """ See :method:`GaugeGroupElement.to_vector` """
        return _np.empty(0, 'd')

    def from_vector(self, v):
        """ See :method:`GaugeGroupElement.from_vector` """
        assert(len(v) == 0)

    def num_params(self):
        """ See :method:`GaugeGroupElement.num_params` """
        return 0
