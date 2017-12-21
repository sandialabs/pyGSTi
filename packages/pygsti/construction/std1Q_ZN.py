from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Variables for working with the a gate set containing Idle, Z(pi/2) and rot(X=pi/2, Y=sqrt(3)/2) gates.
"""

from . import gatestringconstruction as _strc
from . import gatesetconstruction as _setc


gs_target = _setc.build_gateset([2],[('Q0',)], ['Gz','Gn'],
                                [ "Z(pi/2,Q0)", "N(pi/2, sqrt(3)/2, 0, -0.5, Q0)"])

prepStrs = _strc.gatestring_list([(),
                                       ('Gn',),
                                       ('Gn','Gn'),
                                       ('Gn','Gz','Gn'),
                                       ('Gn','Gn','Gn',),
                                       ('Gn','Gz','Gn','Gn','Gn')]) # for 1Q MUB

effectStrs = _strc.gatestring_list([(),
                                       ('Gn',),
                                       ('Gn','Gn'),
                                       ('Gn','Gz','Gn'),
                                       ('Gn','Gn','Gn',),
                                       ('Gn','Gn','Gn','Gz','Gn')]) # for 1Q MUB

germs = _strc.gatestring_list([ ('Gz',),
                                ('Gn',),
                                ('Gz','Gn'),
                                ('Gz','Gz','Gn'),
                                ('Gz','Gn','Gn'),
                                ('Gz','Gz','Gn','Gz','Gn','Gn') ])
germs_lite = germs[:] #same list!


global_fidPairs =  [
    (0, 0), (2, 3), (5, 2), (5, 4)]

pergerm_fidPairsDict = {
  ('Gz',): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
  ('Gn',): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
  ('Gz', 'Gn'): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
  ('Gz', 'Gn', 'Gn'): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
  ('Gz', 'Gz', 'Gn'): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
  ('Gz', 'Gz', 'Gn', 'Gz', 'Gn', 'Gn'): [
        (0, 0), (0, 2), (1, 1), (4, 0), (4, 2), (5, 5)],
}


global_fidPairs_lite =  [
    (0, 0), (2, 3), (5, 2), (5, 4)]

pergerm_fidPairsDict_lite = {
  ('Gz',): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
  ('Gn',): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
  ('Gz', 'Gn'): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
  ('Gz', 'Gn', 'Gn'): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
  ('Gz', 'Gz', 'Gn'): [
        (0, 0), (2, 3), (5, 2), (5, 4)],
  ('Gz', 'Gz', 'Gn', 'Gz', 'Gn', 'Gn'): [
        (0, 0), (0, 2), (1, 1), (4, 0), (4, 2), (5, 5)],
}
