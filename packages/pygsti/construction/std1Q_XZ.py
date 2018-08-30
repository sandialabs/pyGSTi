from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Variables for working with the a gate set containing X(pi/2) and Z(pi/2) gates.
"""

import sys as _sys
from . import gatestringconstruction as _strc
from . import gatesetconstruction as _setc
from . import stdtarget as _stdtarget

description = "X(pi/2) and Z(pi/2) gates"

gates = ['Gx','Gz']
prepStrs = _strc.gatestring_list([(),
                                  ('Gx',),
                                  ('Gx','Gz'),
                                  ('Gx','Gx'),
                                  ('Gx','Gx','Gx'),
                                  ('Gx','Gz','Gx','Gx')]) # for 1Q MUB

effectStrs = _strc.gatestring_list([(),
                                  ('Gx',),
                                  ('Gz','Gx'),
                                  ('Gx','Gx'),
                                  ('Gx','Gx','Gx'),
                                  ('Gx','Gx','Gz','Gx')])

germs = _strc.gatestring_list(
    [ ('Gx',),
      ('Gz',),
      ('Gx','Gz',),
      ('Gx','Gx','Gz'),
      ('Gx','Gz','Gz'),
      ('Gx','Gx','Gz','Gx','Gz','Gz',)])
germs_lite = germs[0:4]


germs = _strc.gatestring_list( [('Gx',), ('Gz',), ('Gz','Gx','Gx'), ('Gz','Gz','Gx')] )

#Construct a target gateset:  X(pi/2), Z(pi/2)
gs_target = _setc.build_gateset([2],[('Q0',)], ['Gx','Gz'],
                                [ "X(pi/2,Q0)", "Z(pi/2,Q0)"])

_gscache = { ("full","auto"): gs_target }
def copy_target(parameterization_type="full", sim_type="auto"):
    """ 
    Returns a copy of the target gateset in the given parameterization.

    Parameters
    ----------
    parameterization_type : {"TP", "CPTP", "H+S", "S", ... }
        The gate and SPAM vector parameterization type. See 
        :function:`GateSet.set_all_parameterizations` for all allowed values.
        
    sim_type : {"auto", "matrix", "map", "termorder:X" }
        The simulator type to be used for gate set calculations (leave as
        "auto" if you're not sure what this is).
    
    Returns
    -------
    GateSet
    """
    return _stdtarget._copy_target(_sys.modules[__name__],parameterization_type,
                                   sim_type, _gscache)


global_fidPairs =  [
    (0, 1), (1, 2), (4, 3), (4, 4)]

pergerm_fidPairsDict = {
  ('Gx',): [
        (1, 1), (3, 4), (4, 2), (5, 5)],
  ('Gz',): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
  ('Gz', 'Gz', 'Gx'): [
        (0, 3), (1, 2), (2, 5), (3, 1), (3, 3), (5, 3)],
  ('Gz', 'Gx', 'Gx'): [
        (0, 3), (0, 4), (1, 0), (1, 4), (2, 1), (4, 5)],
}


global_fidPairs_lite =  [
    (0, 1), (1, 2), (4, 3), (4, 4)]

pergerm_fidPairsDict_lite = {
  ('Gx',): [
        (1, 1), (3, 4), (4, 2), (5, 5)],
  ('Gz',): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
  ('Gx', 'Gz'): [
        (0, 3), (3, 2), (4, 0), (5, 3)],
  ('Gx', 'Gx', 'Gz'): [
        (0, 0), (0, 2), (1, 1), (4, 0), (4, 2), (5, 5)],
}
