""" Functions for allowing old-vesion objects to unpickle load."""
from __future__ import division, print_function, absolute_import #, unicode_literals (don't work w/ModuleType)
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import sys as _sys
import numpy as _np
from types import ModuleType as _ModuleType
from contextlib import contextmanager as _contextmanager

from .. import objects as _objs
from .. import baseobjs as _baseobjs
from ..objects import circuit as _circuit

@_contextmanager
def enable_old_object_unpickling():
    """
    Returns a context manager which enables the unpickling of old-version (0.9.6
    and sometimes prior) objects.
    """
    class dummy_GateString(object):
        def __new__(cls):
            replacement_obj = _circuit.Circuit.__new__(_circuit.Circuit)
            return replacement_obj
    def GateString_setstate(self,state):
        s = state['_str'] if '_str' in state else state['str']
        c = _objs.Circuit(state['_tup'], stringrep=s)
        self.__dict__.update(c.__dict__)

    class dummy_CompressedGateString(object):
        def __new__(cls):
            replacement_obj = _circuit.CompressedCircuit.__new__(_circuit.CompressedCircuit)
            return replacement_obj

    class dummy_GateSet(object):
        def __new__(cls):
            replacement_obj = _objs.ExplicitOpModel.__new__(_objs.ExplicitOpModel)
            return replacement_obj

    class dummy_GateMatrixCalc(object):
        def __new__(cls):
            replacement_obj = _objs.MatrixForwardSimulator.__new__(_objs.MatrixForwardSimulator)
            return replacement_obj

    class dummy_AutoGator(object): pass
    class dummy_SimpleCompositionAutoGator(object): pass

    class dummy_LindbladParameterizedGate(object):
        def __new__(cls):
            replacement_obj = _objs.LindbladDenseOp.__new__(_objs.LindbladDenseOp)
            return replacement_obj
    def Lind_setstate(self,state):
        assert(not state['sparse']), "Can only unpickle old *dense* LindbladParameterizedGate objects"
        g = _objs.LindbladDenseOp.from_operation_matrix(state['base'], state['unitary_postfactor'],
                                                        ham_basis=state['ham_basis'], nonham_basis=state['other_basis'],
                                                        param_mode=state['param_mode'], nonham_mode=state['nonham_mode'],
                                                        truncate=True, mxBasis=state['matrix_basis'],
                                                        evotype=state['_evotype'])
        self.__dict__.update(g.__dict__)

    def Basis_setstate(self,state):
        if "labels" in state: # .label was replaced with ._label
            state['_labels'] = state['labels']
            del state['labels']
        self.__dict__.update(state)

    def Dim_setstate(self,state):
        if "gateDim" in state: # .label was replaced with ._label
            state['opDim'] = state['gateDim']
            del state['gateDim']
        self.__dict__.update(state)

    def ModelMember_setstate(self,state):
        if "dirty" in state: # .dirty was replaced with ._dirty
            state['_dirty'] = state['dirty']
            del state['dirty']
        self.__dict__.update(state)

        
            
    #Modules
    gatestring = _ModuleType("gatestring")
    gatestring.GateString = dummy_GateString
    gatestring.CompressedGateString = dummy_CompressedGateString
    _sys.modules['pygsti.objects.gatestring'] = gatestring
    _objs.circuit.Circuit.__setstate__ = GateString_setstate

    gateset = _ModuleType("gateset")
    gateset.GateSet = dummy_GateSet
    _sys.modules['pygsti.objects.gateset'] = gateset

    gate = _ModuleType("gate")
    gate.EigenvalueParameterizedGate = _objs.EigenvalueParamDenseOp
    gate.LinearlyParameterizedGate = _objs.LinearlyParamDenseOp
    #gate.LindbladParameterizedGateMap = _objs.LindbladOp # no upgrade code for this yet
    gate.LindbladParameterizedGate = dummy_LindbladParameterizedGate
    _objs.LindbladDenseOp.__setstate__ = Lind_setstate #dummy_LindbladParameterizedGate.__setstate__
    gate.FullyParameterizedGate = _objs.FullDenseOp
    gate.TPParameterizedGate = _objs.TPDenseOp
    gate.GateMatrix = _objs.DenseOperator
    gate.ComposedGateMap = _objs.ComposedOp
    gate.EmbeddedGateMap = _objs.EmbeddedOp
    gate.ComposedGate = _objs.ComposedDenseOp
    gate.EmbeddedGate = _objs.EmbeddedDenseOp
    gate.StaticGate = _objs.StaticDenseOp
    gate.LinearlyParameterizedElementTerm = _objs.operation.LinearlyParameterizedElementTerm
    #MapOp = _objs.MapOperator
    _sys.modules['pygsti.objects.gate'] = gate
    
    # spamvec = _ModuleType("spamvec") #already exists - just add to it
    spamvec = _sys.modules['pygsti.objects.spamvec']
    spamvec.LindbladParameterizedSPAMVec = _objs.LindbladSPAMVec
    spamvec.FullyParameterizedSPAMVec = _objs.FullSPAMVec
    spamvec.CPTPParameterizedSPAMVec = _objs.CPTPSPAMVec
    spamvec.TPParameterizedSPAMVec = _objs.TPSPAMVec

    povm = _sys.modules['pygsti.objects.povm']
    povm.LindbladParameterizedPOVM = _objs.LindbladPOVM

    #Don't need class logic here b/c we just store the class itself in a model object:
    gatematrixcalc = _ModuleType("gatematrixcalc")
    gatematrixcalc.GateMatrixCalc = _objs.matrixforwardsim.MatrixForwardSimulator # dummy_GateMatrixCalc
    _sys.modules['pygsti.objects.gatematrixcalc'] = gatematrixcalc

    autogator = _ModuleType("autogator")
    autogator.AutoGator = dummy_AutoGator
    autogator.SimpleCompositionAutoGator = dummy_SimpleCompositionAutoGator
    _sys.modules['pygsti.objects.autogator'] = autogator

    gatestringstructure = _ModuleType("gatestringstructure")
    gatestringstructure.GatestringPlaquette = _objs.circuitstructure.CircuitPlaquette
    gatestringstructure.GateStringStructure = _objs.CircuitStructure
    gatestringstructure.LsGermsStructure = _objs.LsGermsStructure

    _sys.modules['pygsti.objects.gatestringstructure'] = gatestringstructure

    _baseobjs.basis.Basis.__setstate__ = Basis_setstate
    _baseobjs.dim.Dim.__setstate__ = Dim_setstate
    _objs.modelmember.ModelMember.__setstate__ = ModelMember_setstate

    yield # body of context-manager block

#def disable_old_object_unpickling():
#    """
#    Disables the unpickling of old-version (0.9.6 and sometimes prior) objects,
#    which was enabled using :function:`enable_old_object_unpickling`.
#    """
    del _sys.modules['pygsti.objects.gatestring']
    del _sys.modules['pygsti.objects.gateset']
    del _sys.modules['pygsti.objects.gate']
    del _sys.modules['pygsti.objects.gatematrixcalc']
    del _sys.modules['pygsti.objects.autogator']
    del _sys.modules['pygsti.objects.gatestringstructure']

    del _sys.modules['pygsti.objects.spamvec'].LindbladParameterizedSPAMVec
    del _sys.modules['pygsti.objects.spamvec'].FullyParameterizedSPAMVec
    del _sys.modules['pygsti.objects.spamvec'].CPTPParameterizedSPAMVec
    del _sys.modules['pygsti.objects.spamvec'].TPParameterizedSPAMVec

    del _sys.modules['pygsti.objects.povm'].LindbladParameterizedPOVM


    delattr(_objs.Circuit,'__setstate__')
    delattr(_objs.LindbladDenseOp,'__setstate__')
    delattr(_baseobjs.Basis,'__setstate__')
    delattr(_baseobjs.Dim,'__setstate__')
    delattr(_objs.modelmember.ModelMember,'__setstate__')
