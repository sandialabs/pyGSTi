from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Variables for working with the a gate set containing X(pi/2) and Z(pi/2) gates.
"""

from . import gatestringconstruction as _strc
from . import gatesetconstruction as _setc

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

germs = _strc.gatestring_list( [('Gx',), ('Gz',), ('Gz','Gx','Gx'), ('Gz','Gz','Gx')] )

#Construct a target gateset:  X(pi/2), Z(pi/2)
gs_target = _setc.build_gateset([2],[('Q0',)], ['Gx','Gz'],
                                [ "X(pi/2,Q0)", "Z(pi/2,Q0)"],
                                prepLabels=["rho0"], prepExpressions=["0"],
                                effectLabels=["E0"], effectExpressions=["0"],
                                spamdefs={'0': ('rho0','E0'), '1': ('rho0','remainder') },
                                basis='pp')


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
