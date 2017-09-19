from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Variables for working with the a gate set containing Idle, X(pi/2), Y(pi/2), and Z(pi/2) gates.
"""

from . import gatestringconstruction as _strc
from . import gatesetconstruction as _setc

description = "Idle, X(pi/2), Y(pi/2), Z(pi/2) gates"

gates = ['Gi','Gx','Gy', 'Gz']
fiducials = _strc.gatestring_list( [ (), ('Gx',), ('Gy',), ('Gx','Gx'),
                                     ('Gx','Gx','Gx'), ('Gy','Gy','Gy') ] ) # for 1Q MUB
prepStrs = effectStrs = fiducials

germs = _strc.gatestring_list( [('Gi',), ('Gx',), ('Gy',), ('Gz',), ('Gx','Gx','Gy'),
                                ('Gx','Gx','Gz'), ('Gx','Gy','Gy'), ('Gx','Gy','Gz'),
                                ('Gx','Gz','Gz'), ('Gy','Gy','Gz'), ('Gy','Gz','Gz'),
                                ('Gi','Gi','Gi','Gi','Gz'), ('Gi','Gi','Gi','Gi','Gi','Gx'),
                                ('Gi','Gi','Gi','Gi','Gx','Gz'), ('Gi','Gi','Gi','Gi','Gy','Gz')] )

#Construct a target gateset: Identity, X(pi/2), Y(pi/2)
gs_target = _setc.build_gateset([2],[('Q0',)], ['Gi','Gx','Gy','Gz'],
                                [ "I(Q0)","X(pi/2,Q0)", "Y(pi/2,Q0)", "Z(pi/2,Q0)"],
                                 prepLabels=["rho0"], prepExpressions=["0"],
                                 effectLabels=["E0"], effectExpressions=["0"],
                                 spamdefs={'0': ('rho0','E0'), '1': ('rho0','remainder') },
                                 basis='pp')

global_fidPairs =  [
    (0, 1), (1, 2), (4, 3), (4, 4)]

pergerm_fidPairsDict = {
  ('Gx',): [
        (1, 1), (3, 4), (4, 2), (5, 5)],
  ('Gi',): [
        (0, 3), (1, 1), (5, 5)],
  ('Gy',): [
        (0, 2), (2, 2), (2, 4), (4, 4)],
  ('Gz',): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
  ('Gx', 'Gz', 'Gz'): [
        (0, 3), (1, 2), (2, 5), (3, 1), (3, 3), (5, 3)],
  ('Gy', 'Gz', 'Gz'): [
        (1, 3), (1, 4), (3, 5), (5, 0), (5, 4), (5, 5)],
  ('Gy', 'Gy', 'Gz'): [
        (0, 0), (0, 2), (1, 1), (4, 0), (4, 2), (5, 5)],
  ('Gx', 'Gy', 'Gz'): [
        (0, 2), (2, 2), (2, 4), (4, 4)],
  ('Gx', 'Gx', 'Gz'): [
        (0, 0), (0, 4), (1, 5), (2, 3), (2, 5), (5, 5)],
  ('Gx', 'Gx', 'Gy'): [
        (1, 3), (1, 4), (3, 5), (5, 0), (5, 4), (5, 5)],
  ('Gx', 'Gy', 'Gy'): [
        (0, 3), (1, 2), (2, 5), (3, 1), (3, 3), (5, 3)],
  ('Gi', 'Gi', 'Gi', 'Gi', 'Gz'): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
  ('Gi', 'Gi', 'Gi', 'Gi', 'Gi', 'Gx'): [
        (1, 1), (3, 4), (4, 2), (5, 5)],
  ('Gi', 'Gi', 'Gi', 'Gi', 'Gy', 'Gz'): [
        (0, 3), (3, 2), (4, 0), (5, 3)],
  ('Gi', 'Gi', 'Gi', 'Gi', 'Gx', 'Gz'): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
}
