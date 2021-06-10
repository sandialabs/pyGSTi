"""
Functions for allowing old-vesion objects to unpickle load.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


#These functions no longer work, and the changes have become too great to retain
# backward compatibility with old versions.  Keep this commented code around
# for potentially adding similar functionality in future versions.
#
#import numbers as _numbers
#import sys as _sys
#from contextlib import contextmanager as _contextmanager
#from types import ModuleType as _ModuleType
#
#from .. import objects as _objs
#from .. import circuits as _circuits
#from ..circuits import circuit as _circuit
#from ..objects.replib import slowreplib as _slow
#
#
#@_contextmanager
#def enable_no_cython_unpickling():
#    """
#    Context manager for unpickling objects constructed *with* Cython extensions.
#
#    A context manager enabling the un-pickling of pyGSTi objects that
#    were constructed on a system *with* pyGSTi's C-extensions, when the
#    current system's pyGSTi does not have these extensions.
#    """
#
#    class dummy_DMStateRep(object):
#        def __new__(cls, data, reducefix):
#            #replacement_obj = _slow.DMStateRep.__new__(_slow.DMStateRep)
#            replacement_obj = _slow.DMStateRep(data, reducefix)
#            return replacement_obj
#
#    class dummy_DMEffectRepDense(object):
#        def __new__(cls, data, reducefix):
#            #replacement_obj = _slow.DMEffectRepDense.__new__(_slow.DMEffectRepDense)
#            replacement_obj = _slow.DMEffectRepDense(data, reducefix)
#            return replacement_obj
#
#    class dummy_DMOpRepDense(object):
#        def __new__(cls, data, reducefix):
#            #replacement_obj = _slow.DMOpRepDense.__new__(_slow.DMEffectRepDense)
#            replacement_obj = _slow.DMOpRepDense(data, reducefix)
#            return replacement_obj
#
#    assert(_sys.modules.get('pygsti.objects.replib.fastreplib', None) is None), \
#        "You should only use this function when they Cython extensions are *not* built!"
#    fastreplib = _ModuleType("fastreplib")
#    fastreplib.DMStateRep = dummy_DMStateRep
#    fastreplib.DMEffectRepDense = dummy_DMEffectRepDense
#    fastreplib.DMOpRepDense = dummy_DMOpRepDense
#    _sys.modules['pygsti.objects.replib.fastreplib'] = fastreplib
#
#    yield
#
#    del _sys.modules['pygsti.objects.replib.fastreplib']
#
#
#@_contextmanager
#def enable_old_object_unpickling(old_version="0.9.6"):
#    """
#    Context manager enabling unpickling of old-verion objects.
#
#    Returns a context manager which enables the unpickling of old-version (
#    back to 0.9.6 and sometimes prior) objects.
#
#    Parameters
#    ----------
#    old_version : str, optional
#        The string representation of the old version whose pickle files you'd
#        like to unpickle. E.g., `"0.9.7"`
#    """
#    def totup(v): return tuple(map(int, v.split('.')))
#    old_version = totup(old_version)
#
#    if old_version < totup("0.9.6"):
#        raise ValueError(("Cannot unpickle files from version < 0.9.6 with this version."
#                          "  Revert back to 0.9.6 and update to 0.9.6 first."))
#
#    if old_version == totup("0.9.6"):
#        class dummy_GateString(object):
#            def __new__(cls):
#                replacement_obj = _circuit.Circuit.__new__(_circuit.Circuit)
#                return replacement_obj
#
#        def GateString_setstate(self, state):
#            s = state['_str'] if '_str' in state else state['str']
#            c = _objs.Circuit(state['_tup'], stringrep=s)
#            #OLD: self.__dict__.update(c.__dict__)
#            return c.__dict__  # now just return updated Circuit state
#
#        class dummy_CompressedGateString(object):
#            def __new__(cls):
#                replacement_obj = _circuit.CompressedCircuit.__new__(_circuit.CompressedCircuit)
#                return replacement_obj
#
#        class dummy_GateSet(object):
#            def __new__(cls):
#                replacement_obj = _objs.ExplicitOpModel.__new__(_objs.ExplicitOpModel)
#                return replacement_obj
#
#        class dummy_GateMatrixCalc(object):
#            def __new__(cls):
#                replacement_obj = _objs.MatrixForwardSimulator.__new__(_objs.MatrixForwardSimulator)
#                return replacement_obj
#
#        class dummy_AutoGator(object): pass
#        class dummy_SimpleCompositionAutoGator(object): pass
#
#        class dummy_LindbladParameterizedGate(object):
#            def __new__(cls):
#                replacement_obj = _objs.LindbladDenseOp.__new__(_objs.LindbladDenseOp)
#                return replacement_obj
#
#        def Lind_setstate(self, state):
#            assert(not state['sparse']), "Can only unpickle old *dense* LindbladParameterizedGate objects"
#            g = _objs.LindbladDenseOp.from_operation_matrix(state['base'], state['unitary_postfactor'],
#                                                            ham_basis=state['ham_basis'],
#                                                            nonham_basis=state['other_basis'],
#                                                            param_mode=state['param_mode'],
#                                                            nonham_mode=state['nonham_mode'], truncate=True,
#                                                            mx_basis=state['matrix_basis'], evotype=state['_evotype'])
#            self.__dict__.update(g.__dict__)
#
#        def ModelMember_setstate(self, state):
#            if "dirty" in state:  # .dirty was replaced with ._dirty
#                state['_dirty'] = state['dirty']
#                del state['dirty']
#            self.__dict__.update(state)
#
#        #Modules
#        gatestring = _ModuleType("gatestring")
#        gatestring.GateString = dummy_GateString
#        gatestring.CompressedGateString = dummy_CompressedGateString
#        _sys.modules['pygsti.objects.gatestring'] = gatestring
#        #_objs.circuit.Circuit.__setstate__ = GateString_setstate Never needed now
#
#        gateset = _ModuleType("gateset")
#        gateset.GateSet = dummy_GateSet
#        _sys.modules['pygsti.objects.gateset'] = gateset
#
#        gate = _ModuleType("gate")
#        gate.EigenvalueParameterizedGate = _objs.EigenvalueParamDenseOp
#        gate.LinearlyParameterizedGate = _objs.LinearlyParamDenseOp
#        #gate.LindbladParameterizedGateMap = _objs.LindbladOp # no upgrade code for this yet
#        gate.LindbladParameterizedGate = dummy_LindbladParameterizedGate
#        _objs.LindbladDenseOp.__setstate__ = Lind_setstate  # dummy_LindbladParameterizedGate.__setstate__
#        gate.FullyParameterizedGate = _objs.FullDenseOp
#        gate.TPParameterizedGate = _objs.TPDenseOp
#        gate.GateMatrix = _objs.DenseOperator
#        gate.ComposedGateMap = _objs.ComposedOp
#        gate.EmbeddedGateMap = _objs.EmbeddedOp
#        gate.ComposedGate = _objs.ComposedDenseOp
#        gate.EmbeddedGate = _objs.EmbeddedDenseOp
#        gate.StaticGate = _objs.StaticDenseOp
#        gate.LinearlyParameterizedElementTerm = _objs.operation.LinearlyParameterizedElementTerm
#        #MapOp = _objs.MapOperator
#        _sys.modules['pygsti.objects.gate'] = gate
#
#        # spamvec = _ModuleType("spamvec") #already exists - just add to it
#        spamvec = _sys.modules['pygsti.objects.spamvec']
#        spamvec.LindbladParameterizedSPAMVec = _objs.LindbladSPAMVec
#        spamvec.FullyParameterizedSPAMVec = _objs.FullSPAMVec
#        spamvec.CPTPParameterizedSPAMVec = _objs.CPTPSPAMVec
#        spamvec.TPParameterizedSPAMVec = _objs.TPSPAMVec
#
#        povm = _sys.modules['pygsti.objects.povm']
#        povm.LindbladParameterizedPOVM = _objs.LindbladPOVM
#
#        #Don't need class logic here b/c we just store the class itself in a model object:
#        gatematrixcalc = _ModuleType("gatematrixcalc")
#        gatematrixcalc.GateMatrixCalc = _objs.matrixforwardsim.MatrixForwardSimulator  # dummy_GateMatrixCalc
#        _sys.modules['pygsti.objects.gatematrixcalc'] = gatematrixcalc
#
#        autogator = _ModuleType("autogator")
#        autogator.AutoGator = dummy_AutoGator
#        autogator.SimpleCompositionAutoGator = dummy_SimpleCompositionAutoGator
#        _sys.modules['pygsti.objects.autogator'] = autogator
#
#        #These have been removed now!
#        #gatestringstructure = _ModuleType("gatestringstructure")
#        #gatestringstructure.GatestringPlaquette = _objs.circuitstructure.CircuitPlaquette
#        #gatestringstructure.GateStringStructure = _objs.CircuitStructure
#        #gatestringstructure.LsGermsStructure = _objs.LsGermsStructure
#
#        #_sys.modules['pygsti.objects.gatestringstructure'] = gatestringstructure
#
#        _objs.modelmember.ModelMember.__setstate__ = ModelMember_setstate
#
#    if old_version <= totup("0.9.7.1"):
#        class dummy_Basis(object):
#            def __new__(cls):
#                replacement_obj = _objs.basis.BuiltinBasis.__new__(_objs.basis.BuiltinBasis)
#                return replacement_obj
#
#            def __setstate__(self, state):
#                return Basis_setstate(self, state)
#
#        def Basis_setstate(self, state):
#            if "labels" in state:  # .label was replaced with ._label
#                state['_labels'] = state['labels']
#                del state['labels']
#
#            if "name" in state and state['name'] in ('pp', 'std', 'gm', 'qt', 'unknown') and 'dim' in state:
#                dim = state['dim'].opDim if hasattr(state['dim'], 'opDim') else state['dim']
#                assert(isinstance(dim, _numbers.Integral))
#                sparse = state['sparse'] if ('sparse' in state) else False
#                newBasis = _objs.BuiltinBasis(state['name'], int(dim), sparse)
#                self.__class__ = _objs.basis.BuiltinBasis
#                self.__dict__.update(newBasis.__dict__)
#            else:
#                raise ValueError("Can only load old *builtin* basis objects!")
#
#        class dummy_Dim(object):
#            def __setstate__(self, state):  # was Dim_setstate
#                if "gateDim" in state:  # .label was replaced with ._label
#                    state['opDim'] = state['gateDim']
#                    del state['gateDim']
#                self.__dict__.update(state)
#
#        def StateSpaceLabels_setstate(self, state):
#            squared_labeldims = {k: int(d**2) for k, d in state['labeldims'].items()}
#            squared_dims = [tuple((squared_labeldims[lbl] for lbl in tpbLbls))
#                            for tpbLbls in state['labels']]
#            sslbls = _objs.StateSpaceLabels(state['labels'], squared_dims)
#            self.__dict__.update(sslbls.__dict__)
#
#            #DEBUG!!!
#            #print("!!setstate:")
#            #print(state)
#            #assert(False),"STOP"
#
#        def Circuit_setstate(self, state):
#            if old_version == totup("0.9.6"):  # b/c this clobbers older-version upgrade
#                state = GateString_setstate(self, state)
#
#            if 'line_labels' in state: line_labels = state['line_labels']
#            elif '_line_labels' in state: line_labels = state['_line_labels']
#            else: raise ValueError("Cannot determing line labels from old Circuit state: %s" % str(state.keys()))
#
#            if state['_str']:  # then rely on string rep to init new circuit
#                c = _objs.Circuit(None, line_labels, editable=not state['_static'], stringrep=state['_str'])
#            else:
#
#                if 'labels' in state: labels = state['labels']
#                elif '_labels' in state: labels = state['_labels']
#                else: raise ValueError("Cannot determing labels from old Circuit state: %s" % str(state.keys()))
#                c = _objs.Circuit(labels, line_labels, editable=not state['_static'])
#
#            self.__dict__.update(c.__dict__)
#
#        def Hack_CompressedCircuit_expand(self):
#            """ Hacked version to rely on string rep & re-parse if it's there """
#            return _objs.Circuit(None, self._line_labels, editable=False, stringrep=self._str)
#
#        def SPAMVec_setstate(self, state):
#            if "dirty" in state:  # backward compat: .dirty was replaced with ._dirty in ModelMember
#                state['_dirty'] = state['dirty']; del state['dirty']
#            self.__dict__.update(state)
#
#        dim = _ModuleType("dim")
#        dim.Dim = dummy_Dim
#        _sys.modules['pygsti.baseobjs.dim'] = dim
#
#        #_objs.basis.saved_Basis = _objs.basis.Basis
#        #_objs.basis.Basis = dummy_Basis
#        _objs.basis.Basis.__setstate__ = Basis_setstate
#        _circuits.circuit.Circuit.__setstate__ = Circuit_setstate
#        _objs.labeldicts.StateSpaceLabels.__setstate__ = StateSpaceLabels_setstate
#        _circuits.circuit.CompressedCircuit.saved_expand = pygsti.circuits.circuit.CompressedCircuit.expand
#        _circuits.circuit.CompressedCircuit.expand = Hack_CompressedCircuit_expand
#        _objs.spamvec.SPAMVec.__setstate__ = SPAMVec_setstate
#
#    if old_version < totup("0.9.9"):
#
#        def SPAMVec_setstate(self, state):
#            #Note: include "dirty"
#            if old_version <= totup("0.9.7.1"):  # b/c this clobbers older-version upgrade
#                if "dirty" in state:  # backward compat: .dirty was replaced with ._dirty in ModelMember
#                    state['_dirty'] = state['dirty']; del state['dirty']
#            if "_prep_or_effect" not in state:
#                state['_prep_or_effect'] = "unknown"
#            if "base1D" not in state and 'base' in state:
#                state['base1D'] = state['base'].flatten()
#                del state['base']
#
#            self.__dict__.update(state)
#
#        #HERE TODO: need to remake/add ._reps to all spam & operation objects
#
#        _objs.spamvec.SPAMVec.__setstate__ = SPAMVec_setstate
#
#        # Compatibility with refactored `baseobjs` API
#        _sys.modules['pygsti.baseobjs.smartcache'] = _objs.smartcache
#        _sys.modules['pygsti.baseobjs.verbosityprinter'] = _objs.verbosityprinter
#        _sys.modules['pygsti.baseobjs.profiler'] = pygsti.baseobjs.profiler
#        _sys.modules['pygsti.baseobjs.protectedarray'] = _objs.protectedarray
#        _sys.modules['pygsti.baseobjs.objectivefns'] = pygsti.objectivefns.objectivefns
#        _sys.modules['pygsti.baseobjs.basis'] = _objs.basis
#        _sys.modules['pygsti.baseobjs.label'] = _objs.label
#
#    if old_version < totup("0.9.9.1"):
#
#        def DenseOperator_setstate(self, state):
#            if "base" in state:
#                del state['base']
#            self.__dict__.update(state)
#
#        def DenseSPAMVec_setstate(self, state):
#            if old_version <= totup("0.9.9"):  # b/c this clobbers (or shadows) older-version upgrade
#                if old_version <= totup("0.9.7.1"):  # b/c this clobbers older-version upgrade
#                    if "dirty" in state:  # backward compat: .dirty was replaced with ._dirty in ModelMember
#                        state['_dirty'] = state['dirty']; del state['dirty']
#                if "_prep_or_effect" not in state:
#                    state['_prep_or_effect'] = "unknown"
#                if "base1D" not in state and 'base' in state:
#                    state['base1D'] = state['base'].flatten()
#                    del state['base']
#
#            if "base" in state:
#                del state['base']
#            if "base1D" in state:
#                del state['base1D']
#            self.__dict__.update(state)
#
#        _objs.spamvec.DenseSPAMVec.__setstate__ = DenseSPAMVec_setstate
#        _objs.operation.DenseOperator.__setstate__ = DenseOperator_setstate
#
#    yield  # body of context-manager block
#
#    if old_version <= totup("0.9.6"):
#        del _sys.modules['pygsti.objects.gatestring']
#        del _sys.modules['pygsti.objects.gateset']
#        del _sys.modules['pygsti.objects.gate']
#        del _sys.modules['pygsti.objects.gatematrixcalc']
#        del _sys.modules['pygsti.objects.autogator']
#        #del _sys.modules['pygsti.objects.gatestringstructure']
#
#        del _sys.modules['pygsti.objects.spamvec'].LindbladParameterizedSPAMVec
#        del _sys.modules['pygsti.objects.spamvec'].FullyParameterizedSPAMVec
#        del _sys.modules['pygsti.objects.spamvec'].CPTPParameterizedSPAMVec
#        del _sys.modules['pygsti.objects.spamvec'].TPParameterizedSPAMVec
#
#        del _sys.modules['pygsti.objects.povm'].LindbladParameterizedPOVM
#
#        delattr(_objs.Circuit, '__setstate__')
#        delattr(_objs.LindbladDenseOp, '__setstate__')
#        delattr(_objs.modelmember.ModelMember, '__setstate__')
#
#    if old_version <= totup("0.9.7.1"):
#        del _sys.modules['pygsti.baseobjs.dim']
#        delattr(_objs.Basis, '__setstate__')
#        delattr(_objs.labeldicts.StateSpaceLabels, '__setstate__')
#        if hasattr(_objs.Circuit, '__setstate__'):  # b/c above block may have already deleted this
#            delattr(_objs.Circuit, '__setstate__')
#        pygsti.circuits.circuit.CompressedCircuit.expand = pygsti.circuits.circuit.CompressedCircuit.saved_expand
#        delattr(pygsti.circuits.circuit.CompressedCircuit, 'saved_expand')
#        delattr(_objs.spamvec.SPAMVec, '__setstate__')
#
#    if old_version < totup("0.9.9"):
#        if hasattr(_objs.spamvec.SPAMVec, '__setstate__'):  # b/c above block may have already deleted this
#            delattr(_objs.spamvec.SPAMVec, '__setstate__')
#
#        del _sys.modules['pygsti.baseobjs.smartcache']
#        del _sys.modules['pygsti.baseobjs.verbosityprinter']
#        del _sys.modules['pygsti.baseobjs.profiler']
#        del _sys.modules['pygsti.baseobjs.protectedarray']
#        del _sys.modules['pygsti.baseobjs.objectivefns']
#        del _sys.modules['pygsti.baseobjs.basis']
#        del _sys.modules['pygsti.baseobjs.label']
#
#    if old_version < totup("0.9.9.1"):
#        delattr(_objs.spamvec.DenseSPAMVec, '__setstate__')
#        delattr(_objs.operation.DenseOperator, '__setstate__')
