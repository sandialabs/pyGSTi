import pygsti
from pygsti.extras import rb
import numpy as np

def test_clifford_compilations():
    
    # Tests the Clifford compilations hard-coded into the various std gatesets. Perhaps this can be
    # automated to run over all the std gatesets that contain a Clifford compilation?
    
    from pygsti.construction import std1Q_Cliffords
    gs_target = std1Q_Cliffords.gs_target
    clifford_group = rb.group.construct_1Q_Clifford_group()

    from pygsti.construction import std1Q_XY
    gs_target = std1Q_XY.gs_target.copy()
    clifford_compilation = std1Q_XY.clifford_compilation
    compiled_cliffords = pygsti.construction.build_alias_gateset(gs_target,clifford_compilation)

    for key in list(compiled_cliffords.gates.keys()):
        assert(np.sum(abs(compiled_cliffords[key]-clifford_group.get_matrix(key))) < 10**(-10))

    from pygsti.construction import std1Q_XYI
    gs_target = std1Q_XYI.gs_target.copy()
    clifford_compilation = std1Q_XYI.clifford_compilation
    compiled_cliffords = pygsti.construction.build_alias_gateset(gs_target,clifford_compilation)

    for key in list(compiled_cliffords.gates.keys()):
        assert(np.sum(abs(compiled_cliffords[key]-clifford_group.get_matrix(key))) < 10**(-10))

    # Future : add the remaining Clifford compilations here.
 