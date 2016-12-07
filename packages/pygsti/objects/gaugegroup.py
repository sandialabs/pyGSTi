from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" GaugeGroup and derived objects, used primarily in gauge optimization """

import numpy as _np

class GaugeGroup(object):
    def __init__(self):
        pass

    def num_params(self):
        return 0

    def get_element(self, param_vec):
        return self.element()

    def get_initial_params(self):
        return _np.array([],'d')

    class element(object):
        def __init__(self): pass
        def get_transform_matrix(self): return None
        def get_transform_matrix_inverse(self): return None
        def to_vector(self): return _np.array([],'d')
        def from_vector(self,v): pass


class GateGaugeGroup(GaugeGroup):
    def __init__(self, gate):
        self.gate = gate

    def num_params(self):
        return self.gate.num_params()

    def get_element(self, param_vec):
        elgate = self.gate.copy()
        elgate.from_vector(param_vec)
        return self.element(elgate)
    
    def get_initial_params(self):
        return self.gate.to_vector()

    class element(GaugeGroup.element):
        def __init__(self, gate):  
            self.gate = gate
            self._inv_matrix = None
            GaugeGroup.element.__init__(self)

        def get_transform_matrix(self): 
            return _np.asarray(self.gate)

        def get_transform_matrix_inverse(self): 
            if self._inv_matrix is None:
                self._inv_matrix = _np.linalg.inv(_np.asarray(self.gate))
            return self._inv_matrix

        def to_vector(self):
            return self.gate.to_vector()

        def from_vector(self,v):
            self.gate.from_vector(v)
            self._inv_matrix = None



class FullGaugeGroup(GateGaugeGroup):
    def __init__(self, dim):
        from . import gate as _gate #b/c gate.py imports gaugegroup
        gate = _gate.FullyParameterizedGate(_np.identity(dim,'d'))
        GateGaugeGroup.__init__(self, gate)

    class element(GateGaugeGroup.element):
        def __init__(self, gate):
            GateGaugeGroup.element.__init__(self,gate)
            pass #inherits everything it needs

class TPGaugeGroup(GateGaugeGroup):
    def __init__(self, dim):
        from . import gate as _gate #b/c gate.py imports gaugegroup
        gate = _gate.TPParameterizedGate(_np.identity(dim,'d'))
        GateGaugeGroup.__init__(self, gate)

    class element(GateGaugeGroup.element):
        def __init__(self, gate):
            GateGaugeGroup.element.__init__(self,gate)

#        pass #inherits everything it needs

class DiagGaugeGroup(GateGaugeGroup):
    def __init__(self, dim):
        from . import gate as _gate #b/c gate.py imports gaugegroup
        ltrans = _np.identity(dim,'d')
        rtrans = _np.identity(dim,'d')
        baseMx = _np.identity(dim,'d')
        parameterArray = _np.zeros(dim, 'd')
        parameterToBaseIndicesMap = { i: [(i,i)] for i in range(dim) }
        gate = _gate.LinearlyParameterizedGate(baseMx, parameterArray,
                                               parameterToBaseIndicesMap,
                                               ltrans, rtrans, real=True)
        GateGaugeGroup.__init__(self, gate)

    class element(GateGaugeGroup.element):
        def __init__(self, gate):
            GateGaugeGroup.element.__init__(self,gate)

#        pass #inherits everything it needs


class TPDiagGaugeGroup(TPGaugeGroup):
    def __init__(self, dim):
        from . import gate as _gate #b/c gate.py imports gaugegroup
        ltrans = _np.identity(dim,'d')
        rtrans = _np.identity(dim,'d')
        baseMx = _np.identity(dim,'d')
        parameterArray = _np.zeros(dim-1, 'd')
        parameterToBaseIndicesMap = { i: [(i+1,i+1)] for i in range(dim-1) }
        gate = _gate.LinearlyParameterizedGate(baseMx, parameterArray,
                                               parameterToBaseIndicesMap,
                                               ltrans, rtrans, real=True)
        GateGaugeGroup.__init__(self, gate)

    class element(TPGaugeGroup.element):
        def __init__(self, gate):
            TPGaugeGroup.element.__init__(self,gate)

#        pass #inherits everything it needs


class UnitaryGaugeGroup(GateGaugeGroup):
    def __init__(self, dim):
        from . import gate as _gate #b/c gate.py imports gaugegroup
        gate = _gate.LindbladParameterizedGate(None, _np.identity(dim,'d'),
                                               cptp=True, nonHamTerms="none")
        GateGaugeGroup.__init__(self, gate)

    class element(GateGaugeGroup.element):
        def __init__(self, gate):
            GateGaugeGroup.element.__init__(self,gate)
