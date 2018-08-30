from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Variables for working with the a gate set containing Idle, X(pi/2), Y(pi/2), and Z(pi/2) gates.
"""

import sys as _sys
from . import gatestringconstruction as _strc
from . import gatesetconstruction as _setc
from . import stdtarget as _stdtarget

description = "Idle, X(pi/2), Y(pi/2), Z(pi/2) gates"

gates = ['Gi','Gx','Gy', 'Gz']
fiducials = _strc.gatestring_list( [ (), ('Gx',), ('Gy',), ('Gx','Gx'),
                                     ('Gx','Gx','Gx'), ('Gy','Gy','Gy') ] ) # for 1Q MUB
prepStrs = effectStrs = fiducials

germs = _strc.gatestring_list( 
    [('Gi',), ('Gx',), ('Gy',), ('Gz',),
     ('Gx','Gz'), ('Gx','Gy'),
     ('Gx','Gx','Gy'), ('Gx','Gx','Gz'),
     ('Gy','Gy','Gz'), ('Gx','Gy','Gz'),
     ('Gx','Gy','Gi'), ('Gx','Gi','Gy'),
     ('Gx','Gi','Gi'), ('Gy','Gi','Gi'),
     ('Gi','Gx','Gz'), ('Gi','Gy','Gz'),
     ('Gx','Gy','Gy','Gi'), ('Gx','Gx','Gy','Gx','Gy','Gy') ])
germs_lite = germs[0:10]

#Construct a target gateset: Identity, X(pi/2), Y(pi/2)
gs_target = _setc.build_gateset([2],[('Q0',)], ['Gi','Gx','Gy','Gz'],
                                [ "I(Q0)","X(pi/2,Q0)", "Y(pi/2,Q0)", "Z(pi/2,Q0)"])

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
    (0, 0), (2, 3), (5, 2), (5, 4)]

pergerm_fidPairsDict = {
  ('Gx',): [
        (1, 1), (3, 4), (4, 2), (5, 5)],
  ('Gi',): [
        (0, 3), (1, 1), (5, 5)],
  ('Gz',): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
  ('Gy',): [
        (0, 2), (2, 2), (2, 4), (4, 4)],
  ('Gx', 'Gz'): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
  ('Gx', 'Gy'): [
        (0, 0), (0, 4), (2, 5), (5, 4)],
  ('Gy', 'Gi', 'Gi'): [
        (0, 2), (2, 2), (2, 4), (4, 4)],
  ('Gx', 'Gx', 'Gz'): [
        (0, 0), (0, 4), (1, 5), (2, 3), (2, 5), (5, 5)],
  ('Gy', 'Gy', 'Gz'): [
        (0, 0), (0, 2), (1, 1), (4, 0), (4, 2), (5, 5)],
  ('Gx', 'Gy', 'Gi'): [
        (0, 0), (0, 4), (2, 5), (5, 4)],
  ('Gx', 'Gy', 'Gz'): [
        (0, 2), (2, 2), (2, 4), (4, 4)],
  ('Gx', 'Gi', 'Gy'): [
        (0, 0), (0, 4), (2, 5), (5, 4)],
  ('Gx', 'Gx', 'Gy'): [
        (1, 3), (1, 4), (3, 5), (5, 0), (5, 4), (5, 5)],
  ('Gi', 'Gx', 'Gz'): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
  ('Gi', 'Gy', 'Gz'): [
        (0, 3), (3, 2), (4, 0), (5, 3)],
  ('Gx', 'Gi', 'Gi'): [
        (1, 1), (3, 4), (4, 2), (5, 5)],
  ('Gx', 'Gy', 'Gy', 'Gi'): [
        (0, 3), (1, 2), (2, 5), (3, 1), (3, 3), (5, 3)],
  ('Gx', 'Gx', 'Gy', 'Gx', 'Gy', 'Gy'): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
}


global_fidPairs_lite =  [
    (0, 4), (0, 5), (1, 0), (2, 0), (2, 4), (2, 5), (3, 0), (4, 2), 
    (4, 4), (5, 1), (5, 2), (5, 3)]

pergerm_fidPairsDict_lite = {
  ('Gx',): [
        (1, 1), (3, 4), (4, 2), (5, 5)],
  ('Gi',): [
        (0, 3), (1, 1), (5, 5)],
  ('Gz',): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
  ('Gy',): [
        (0, 2), (2, 2), (2, 4), (4, 4)],
  ('Gx', 'Gz'): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
  ('Gx', 'Gy'): [
        (0, 0), (0, 4), (2, 5), (5, 4)],
  ('Gy', 'Gy', 'Gz'): [
        (0, 0), (0, 2), (1, 1), (4, 0), (4, 2), (5, 5)],
  ('Gx', 'Gy', 'Gz'): [
        (0, 2), (2, 2), (2, 4), (4, 4)],
  ('Gx', 'Gx', 'Gy'): [
        (1, 3), (1, 4), (3, 5), (5, 0), (5, 4), (5, 5)],
  ('Gx', 'Gx', 'Gz'): [
        (0, 0), (0, 4), (1, 5), (2, 3), (2, 5), (5, 5)],
}
