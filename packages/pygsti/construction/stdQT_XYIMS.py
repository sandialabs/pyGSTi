from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Variables for working with the a qutrit gate set containing Idle, X(pi/2) and Y(pi/2) and Molmer-Sorenson gates.
"""

from . import gatestringconstruction as _strc
from . import gatesetconstruction as _setc
from . import qutrit as _qutrit
from collections import OrderedDict as _OrderedDict
from numpy import pi as _pi

description = "Idle, symmetric X(pi/2), symmetric Y(pi/2), and Molmer-Sorenson gates"

gates = ['Gi','Gx','Gy','Gm']
prepStrs = _strc.gatestring_list([
    (), ('Gx',), ('Gy',), ('Gm',),
    ('Gx','Gx'), ('Gm','Gx'), ('Gm','Gy'),
    ('Gy','Gy','Gy'),('Gx','Gx','Gx') ])

effectStrs = _strc.gatestring_list([
    (),('Gy',),('Gx',), ('Gm',),
    ('Gx','Gx'),('Gy','Gm'),('Gx','Gm') ])

germs = _strc.gatestring_list([
    ('Gi',),
    ('Gx',),
    ('Gy',),
    ('Gm',),
    ('Gx', 'Gy'),
    ('Gx', 'Gm'),
    ('Gx', 'Gy', 'Gi'),
    ('Gx', 'Gi', 'Gy'),
    ('Gx', 'Gi', 'Gi'),
    ('Gy', 'Gi', 'Gi'),
    ('Gx', 'Gy', 'Gm'),
    ('Gx', 'Gm', 'Gy'),
    ('Gi', 'Gx', 'Gm'),
    ('Gy', 'Gm', 'Gm'),
    ('Gx', 'Gy', 'Gy'),
    ('Gi', 'Gm', 'Gx'),
    ('Gx', 'Gx', 'Gi', 'Gy'),
    ('Gx', 'Gy', 'Gy', 'Gi'),
    ('Gy', 'Gm', 'Gm', 'Gm'),
    ('Gy', 'Gy', 'Gm', 'Gm'),
    ('Gx', 'Gm', 'Gy', 'Gx'),
    ('Gx', 'Gm', 'Gm', 'Gm'),
    ('Gx', 'Gi', 'Gy', 'Gy'),
    ('Gy', 'Gx', 'Gy', 'Gi'),
    ('Gx', 'Gy', 'Gm', 'Gy'),
    ('Gm', 'Gm', 'Gi', 'Gi'),
    ('Gy', 'Gx', 'Gy', 'Gm'),
    ('Gi', 'Gx', 'Gm', 'Gx'),
    ('Gx', 'Gx', 'Gy', 'Gx'),
    ('Gx', 'Gi', 'Gm', 'Gi'),
    ('Gm', 'Gy', 'Gm', 'Gx'),
    ('Gx', 'Gx', 'Gy', 'Gy'),
    ('Gm', 'Gy', 'Gm', 'Gi'),
    ('Gi', 'Gx', 'Gy', 'Gm'),
    ('Gm', 'Gi', 'Gx', 'Gi'),
    ('Gy', 'Gy', 'Gy', 'Gy'),
    ('Gi', 'Gy', 'Gy', 'Gm'),
    ('Gy', 'Gy', 'Gx', 'Gx', 'Gy'),
    ('Gm', 'Gi', 'Gm', 'Gy', 'Gi'),
    ('Gy', 'Gi', 'Gi', 'Gy', 'Gx'),
    ('Gx', 'Gy', 'Gm', 'Gy', 'Gy'),
    ('Gx', 'Gi', 'Gm', 'Gi', 'Gy'),
    ('Gy', 'Gm', 'Gx', 'Gy', 'Gy'),
    ('Gx', 'Gy', 'Gy', 'Gy', 'Gy'),
    ('Gm', 'Gy', 'Gm', 'Gm', 'Gy'),
    ('Gx', 'Gy', 'Gm', 'Gx', 'Gi'),
    ('Gx', 'Gx', 'Gy', 'Gm', 'Gy'),
    ('Gm', 'Gx', 'Gi', 'Gx', 'Gx'),
    ('Gy', 'Gi', 'Gm', 'Gx', 'Gi'),
    ('Gy', 'Gy', 'Gx', 'Gm', 'Gx'),
    ('Gm', 'Gx', 'Gi', 'Gy', 'Gx'),
    ('Gx', 'Gx', 'Gy', 'Gx', 'Gy', 'Gy'),
    ('Gx', 'Gi', 'Gi', 'Gy', 'Gy', 'Gy'),
    ('Gm', 'Gm', 'Gi', 'Gi', 'Gy', 'Gi'),
    ('Gy', 'Gx', 'Gx', 'Gy', 'Gx', 'Gm'),
    ('Gi', 'Gm', 'Gx', 'Gy', 'Gm', 'Gy'),
    ('Gm', 'Gy', 'Gi', 'Gx', 'Gy', 'Gi'),
    ('Gi', 'Gx', 'Gx', 'Gi', 'Gy', 'Gy'),
    ('Gy', 'Gi', 'Gx', 'Gx', 'Gy', 'Gm'),
    ('Gm', 'Gx', 'Gy', 'Gx', 'Gx', 'Gx'),
    ('Gi', 'Gy', 'Gx', 'Gx', 'Gy', 'Gy'),
    ('Gm', 'Gy', 'Gx', 'Gm', 'Gm', 'Gy') ])


germs_lite = _strc.gatestring_list([
    ('Gi',),
    ('Gy',),
    ('Gx',),
    ('Gm',),
    ('Gi', 'Gy'),
    ('Gi', 'Gx'),
    ('Gi', 'Gm'),
    ('Gy', 'Gx'),
    ('Gy', 'Gm'),
    ('Gx', 'Gm'),
    ('Gi', 'Gi', 'Gy'),
    ('Gi', 'Gi', 'Gx'),
    ('Gi', 'Gi', 'Gm'),
    ('Gi', 'Gy', 'Gy'),
    ('Gi', 'Gy', 'Gx'),
    ('Gi', 'Gy', 'Gm'),
    ('Gi', 'Gx', 'Gy'),
    ('Gi', 'Gx', 'Gx'),
    ('Gi', 'Gx', 'Gm'),
    ('Gi', 'Gm', 'Gy'),
    ('Gi', 'Gm', 'Gx'),
    ('Gi', 'Gm', 'Gm'),
    ('Gy', 'Gy', 'Gx'),
    ('Gy', 'Gy', 'Gm'),
    ('Gy', 'Gx', 'Gx'),
    ('Gy', 'Gx', 'Gm'),
    ('Gy', 'Gm', 'Gx'),
    ('Gy', 'Gm', 'Gm'),
    ('Gx', 'Gx', 'Gm'),
    ('Gx', 'Gm', 'Gm') ])


#Construct a target gateset: Identity, sym X(pi/2), sym Y(pi/2), Molmer-Sorenson
gs_target = _qutrit.make_qutrit_gateset(errorScale=0, Xangle=_pi/2, Yangle=_pi/2,
                                       MSglobal=_pi/2, MSlocal=0, basis="qt")

legacy_gs_target = _qutrit.make_qutrit_gateset(errorScale=0, Xangle=-_pi/2, Yangle=_pi/2,
                                       MSglobal=-_pi/2, MSlocal=0, basis="qt")
  #Note: negative signs from weird/incorrect conventions
