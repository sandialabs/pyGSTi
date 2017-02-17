from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Variables for working with the 2-qubit gate set containing the gates
I*X(pi/2), I*Y(pi/2), X(pi/2)*I, Y(pi/2)*I, and CNOT.
"""

from . import gatestringconstruction as _strc
from . import gatesetconstruction as _setc
from . import spamspecconstruction as _spamc

description = "I*X(pi/2), I*Y(pi/2), X(pi/2)*I, Y(pi/2)*I, and CNOT gates"

gates = ['Gix','Giy','Gxi','Gyi','Gcnot']

fiducials16 = _strc.gatestring_list(
    [ (), ('Gix',), ('Giy',), ('Gix','Gix'),
      ('Gxi',), ('Gxi','Gix'), ('Gxi','Giy'), ('Gxi','Gix','Gix'),
      ('Gyi',), ('Gyi','Gix'), ('Gyi','Giy'), ('Gyi','Gix','Gix'),
      ('Gxi','Gxi'), ('Gxi','Gxi','Gix'), ('Gxi','Gxi','Giy'), ('Gxi','Gxi','Gix','Gix') ] )

fiducials36 = _strc.gatestring_list(
    [ (), ('Gix',), ('Giy',), ('Gix','Gix'), ('Gix','Gix','Gix'), ('Giy','Giy','Giy'),
      ('Gxi',), ('Gxi','Gix'), ('Gxi','Giy'), ('Gxi','Gix','Gix'), ('Gxi','Gix','Gix','Gix'), ('Gxi','Giy','Giy','Giy'),
      ('Gyi',), ('Gyi','Gix'), ('Gyi','Giy'), ('Gyi','Gix','Gix'), ('Gyi','Gix','Gix','Gix'), ('Gyi','Giy','Giy','Giy'),
      ('Gxi','Gxi'), ('Gxi','Gxi','Gix'), ('Gxi','Gxi','Giy'), ('Gxi','Gxi','Gix','Gix'), ('Gxi','Gxi','Gix','Gix','Gix'),
      ('Gxi','Gxi','Giy','Giy','Giy'), ('Gxi','Gxi','Gxi'), ('Gxi','Gxi','Gxi','Gix'), ('Gxi','Gxi','Gxi','Giy'),
      ('Gxi','Gxi','Gxi','Gix','Gix'), ('Gxi','Gxi','Gxi','Gix','Gix','Gix'), ('Gxi','Gxi','Gxi','Giy','Giy','Giy'),
      ('Gyi','Gyi','Gyi'), ('Gyi','Gyi','Gyi','Gix'), ('Gyi','Gyi','Gyi','Giy'), ('Gyi','Gyi','Gyi','Gix','Gix'),
      ('Gyi','Gyi','Gyi','Gix','Gix','Gix'), ('Gyi','Gyi','Gyi','Giy','Giy','Giy') ] )

fiducials = fiducials16
prepStrs = fiducials16

effectStrs = _strc.gatestring_list(
    [(), ('Gix',), ('Giy',), 
     ('Gix','Gix'), ('Gxi',), 
     ('Gyi',), ('Gxi','Gxi'), 
     ('Gxi','Gix'), ('Gxi','Giy'), 
     ('Gyi','Gix'), ('Gyi','Giy')] )

legacy_effectStrs = _strc.gatestring_list(
    [ (), ('Gix',), ('Giy',), ('Gxi',), ('Gyi',),
      ('Gix','Gxi'), ('Gxi','Giy'), ('Gyi','Gix'),
      ('Gyi','Giy'), ('Gxi','Gxi') ] )

germs = _strc.gatestring_list(
    [('Gxi',), ('Gyi',), ('Gix',), ('Giy',), ('Gcnot',),
     ('Gxi', 'Gyi'), ('Gix', 'Giy'), ('Giy', 'Gyi'), ('Gix', 'Gyi'),
     ('Gyi', 'Gcnot'), ('Giy', 'Gcnot'),
     ('Gxi', 'Gcnot', 'Gcnot'),
     ('Giy', 'Gxi', 'Gcnot'),
     ('Giy', 'Gcnot', 'Gyi'),
     ('Giy', 'Gyi', 'Gcnot'),
     ('Gix', 'Gxi', 'Gcnot'),
     ('Giy', 'Giy', 'Gcnot'),
     ('Giy', 'Gcnot', 'Gxi'),
     ('Gix', 'Giy', 'Gcnot'),
     ('Giy', 'Gxi', 'Gyi'),
     ('Gix', 'Giy', 'Gyi'),
     ('Gyi', 'Gyi', 'Gyi', 'Gxi'),
     ('Giy', 'Giy', 'Giy', 'Gix'),
     ('Gxi', 'Gyi', 'Gix', 'Giy'),
     ('Gcnot', 'Gix', 'Gyi', 'Gyi'),
     ('Gcnot', 'Gix', 'Gix', 'Gcnot'),
     ('Gxi', 'Gcnot', 'Gyi', 'Gyi'),
     ('Gyi', 'Gyi', 'Gyi', 'Gix'),
     ('Gix', 'Gix', 'Giy', 'Gcnot', 'Gcnot'),
     ('Gcnot', 'Giy', 'Giy', 'Gix', 'Giy'),
     ('Gyi', 'Gcnot', 'Gix', 'Giy', 'Gyi'),
     ('Giy', 'Gxi', 'Gcnot', 'Gxi', 'Gcnot'),
     ('Gyi', 'Gcnot', 'Gxi', 'Gcnot', 'Gxi'),
     ('Gyi', 'Gxi', 'Gyi', 'Gxi', 'Gxi', 'Gxi'),
     ('Gyi', 'Gxi', 'Gyi', 'Gyi', 'Gxi', 'Gxi'),
     ('Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gyi', 'Gxi'),
     ('Gxi', 'Gxi', 'Gyi', 'Gxi', 'Gyi', 'Gyi'),
     ('Giy', 'Gix', 'Giy', 'Gix', 'Gix', 'Gix'),
     ('Giy', 'Gix', 'Giy', 'Giy', 'Gix', 'Gix'),
     ('Giy', 'Giy', 'Giy', 'Gix', 'Giy', 'Gix'),
     ('Gix', 'Gix', 'Giy', 'Gix', 'Giy', 'Giy'),
     ('Gcnot', 'Gyi', 'Giy', 'Gxi', 'Gix', 'Gcnot'),
     ('Gxi', 'Giy', 'Gxi', 'Gcnot', 'Gyi', 'Gix'),
     ('Gxi', 'Giy', 'Giy', 'Giy', 'Gcnot', 'Gxi'),
     ('Gcnot', 'Gxi', 'Gcnot', 'Gxi', 'Giy', 'Gix'),
     ('Gyi', 'Gix', 'Gyi', 'Gix', 'Gxi', 'Gxi'),
     ('Gix', 'Gcnot', 'Gxi', 'Gix', 'Gxi', 'Gcnot'),
     ('Gxi', 'Giy', 'Gyi', 'Gxi', 'Gcnot', 'Gcnot'),
     ('Gix', 'Gix', 'Giy', 'Gcnot', 'Giy', 'Gcnot', 'Gxi'),
     ('Giy', 'Gxi', 'Gcnot', 'Gix', 'Gix', 'Giy', 'Giy'),
     ('Gxi', 'Gcnot', 'Giy', 'Gyi', 'Gxi', 'Gix', 'Giy'),
     ('Gcnot', 'Gcnot', 'Gix', 'Gxi', 'Giy', 'Gxi', 'Gxi'),
     ('Gxi', 'Gix', 'Giy', 'Gyi', 'Gix', 'Gix', 'Gix'),
     ('Gxi', 'Gix', 'Gyi', 'Gix', 'Gyi', 'Giy', 'Gyi'),
     ('Gix', 'Gix', 'Gix', 'Gix', 'Gxi', 'Gxi', 'Gyi'),
     ('Giy', 'Gcnot', 'Gxi', 'Gyi', 'Gyi', 'Gcnot', 'Gix', 'Gcnot'),
     ('Gxi', 'Gyi', 'Gxi', 'Giy', 'Gxi', 'Giy', 'Gix', 'Giy'),
     ('Giy', 'Giy', 'Gyi', 'Gix', 'Gcnot', 'Gxi', 'Gyi', 'Gyi'),
     ('Gxi', 'Gix', 'Gcnot', 'Gyi', 'Gix', 'Gcnot', 'Gix', 'Giy'),
     ('Gix', 'Gxi', 'Gxi', 'Giy', 'Gxi', 'Gyi', 'Gix', 'Gcnot'),
     ('Gix', 'Gix', 'Gyi', 'Gxi', 'Giy', 'Gix', 'Gcnot', 'Gyi'),
     ('Gix', 'Giy', 'Gix', 'Gxi', 'Gix', 'Giy', 'Gxi', 'Gxi'),
     ('Giy', 'Gix', 'Gcnot', 'Gxi', 'Gcnot', 'Gxi', 'Gcnot', 'Gyi'),
     ('Gxi', 'Giy', 'Gix', 'Gix', 'Gxi', 'Giy', 'Gxi', 'Gcnot'),
     ('Gyi', 'Gyi', 'Gyi', 'Gyi', 'Gix', 'Giy', 'Gix', 'Gyi')
     ])

legacy_germs = _strc.gatestring_list(
    [('Giy',),
     ('Gxi',),
     ('Gyi',),
     ('Gcnot',),
     ('Gix', 'Gyi'),
     ('Giy', 'Gyi'),
     ('Giy', 'Gcnot'),
     ('Gyi', 'Gcnot'),
     ('Gix', 'Gix', 'Giy'),
     ('Gix', 'Gix', 'Gyi'),
     ('Gix', 'Giy', 'Giy'),
     ('Gix', 'Giy', 'Gyi'),
     ('Gix', 'Giy', 'Gcnot'),
     ('Gix', 'Gxi', 'Gcnot'),
     ('Gix', 'Gcnot', 'Giy'),
     ('Gix', 'Gcnot', 'Gyi'),
     ('Giy', 'Giy', 'Gxi'),
     ('Giy', 'Giy', 'Gcnot'),
     ('Giy', 'Gxi', 'Gyi'),
     ('Giy', 'Gxi', 'Gcnot'),
     ('Giy', 'Gyi', 'Gxi'),
     ('Giy', 'Gyi', 'Gcnot'),
     ('Giy', 'Gcnot', 'Gxi'),
     ('Giy', 'Gcnot', 'Gyi'),
     ('Gxi', 'Gxi', 'Gyi'),
     ('Gxi', 'Gxi', 'Gcnot'),
     ('Gxi', 'Gyi', 'Gyi'),
     ('Gxi', 'Gcnot', 'Gcnot'),
     ('Gcnot', 'Gix', 'Gyi', 'Gyi'),
     ('Gcnot', 'Gix', 'Gix', 'Gcnot'),
     ('Gxi', 'Giy', 'Gix', 'Giy'),
     ('Gxi', 'Gyi', 'Gix', 'Giy'),
     ('Gyi', 'Gyi', 'Gyi', 'Gix'),
     ('Gxi', 'Gcnot', 'Gyi', 'Gyi'),
     ('Gyi', 'Gcnot', 'Gix', 'Giy', 'Gyi'),
     ('Gix', 'Gix', 'Giy', 'Gcnot', 'Gcnot'),
     ('Giy', 'Gxi', 'Gcnot', 'Gxi', 'Gcnot'),
     ('Gyi', 'Gcnot', 'Gxi', 'Gcnot', 'Gxi'),
     ('Gcnot', 'Giy', 'Giy', 'Gix', 'Giy'),
     ('Giy', 'Gix', 'Gix', 'Gyi', 'Gxi'),
     ('Giy', 'Gix', 'Gyi', 'Giy', 'Giy'),
     ('Gcnot', 'Gxi', 'Gcnot', 'Gix', 'Gyi'),
     ('Gix', 'Gix', 'Gyi', 'Gxi', 'Gyi', 'Gix'),
     ('Gxi', 'Giy', 'Gyi', 'Gxi', 'Gcnot', 'Gcnot'),
     ('Gcnot', 'Gyi', 'Gcnot', 'Gxi', 'Gyi', 'Gyi'),
     ('Gxi', 'Giy', 'Gxi', 'Gcnot', 'Gyi', 'Gix'),
     ('Gcnot', 'Gyi', 'Giy', 'Gxi', 'Gix', 'Gcnot'),
     ('Gcnot', 'Gxi', 'Gcnot', 'Gxi', 'Giy', 'Gix'),
     ('Gxi', 'Gxi', 'Giy', 'Gix', 'Giy', 'Gix'),
     ('Gxi', 'Giy', 'Giy', 'Giy', 'Gcnot', 'Gxi'),
     ('Gyi', 'Gix', 'Gyi', 'Gix', 'Gxi', 'Gxi'),
     ('Gix', 'Gcnot', 'Gxi', 'Gix', 'Gxi', 'Gcnot'),
     ('Gxi', 'Gix', 'Giy', 'Gyi', 'Gix', 'Gix', 'Gix'),
     ('Gyi', 'Gxi', 'Gyi', 'Giy', 'Giy', 'Gxi', 'Gix'),
     ('Gcnot', 'Gcnot', 'Gix', 'Gxi', 'Giy', 'Gxi', 'Gxi'),
     ('Gxi', 'Gcnot', 'Giy', 'Gyi', 'Gxi', 'Gix', 'Giy'),
     ('Gxi', 'Gix', 'Gyi', 'Gix', 'Gyi', 'Giy', 'Gyi'),
     ('Gix', 'Gcnot', 'Giy', 'Gcnot', 'Gcnot', 'Gix', 'Gix'),
     ('Giy', 'Gxi', 'Gcnot', 'Gix', 'Gix', 'Giy', 'Giy'),
     ('Gix', 'Gix', 'Gix', 'Gix', 'Gxi', 'Gxi', 'Gyi'),
     ('Gix', 'Gix', 'Giy', 'Gcnot', 'Giy', 'Gcnot', 'Gxi'),
     ('Gxi', 'Giy', 'Gix', 'Gix', 'Gxi', 'Giy', 'Gxi', 'Gcnot'),
     ('Giy', 'Gix', 'Gcnot', 'Gxi', 'Gcnot', 'Gxi', 'Gcnot', 'Gyi'),
     ('Gix', 'Giy', 'Gix', 'Gxi', 'Gix', 'Giy', 'Gxi', 'Gxi'),
     ('Gix', 'Gix', 'Gyi', 'Gxi', 'Giy', 'Gix', 'Gcnot', 'Gyi'),
     ('Gxi', 'Gyi', 'Gxi', 'Giy', 'Gxi', 'Giy', 'Gix', 'Giy'),
     ('Giy', 'Giy', 'Gyi', 'Gix', 'Gcnot', 'Gxi', 'Gyi', 'Gyi'),
     ('Gix', 'Gxi', 'Gxi', 'Giy', 'Gxi', 'Gyi', 'Gix', 'Gcnot'),
     ('Gyi', 'Gyi', 'Gyi', 'Gyi', 'Gix', 'Giy', 'Gix', 'Gyi'),
     ('Gxi', 'Gix', 'Gcnot', 'Gyi', 'Gix', 'Gcnot', 'Gix', 'Giy'),
     ('Giy', 'Gcnot', 'Gxi', 'Gyi', 'Gyi', 'Gcnot', 'Gix', 'Gcnot')] )

#Construct the target gateset
gs_target = _setc.build_gateset(
    [4], [('Q0','Q1')],['Gix','Giy','Gxi','Gyi','Gcnot'],
    [ "I(Q0):X(pi/2,Q1)", "I(Q0):Y(pi/2,Q1)", "X(pi/2,Q0):I(Q1)", "Y(pi/2,Q0):I(Q1)", "CX(pi,Q0,Q1)" ],
    prepLabels=['rho0'], prepExpressions=["0"],
    effectLabels=['E0','E1','E2'], effectExpressions=["0","1","2"],
    spamdefs={'upup': ('rho0','E0'), 'updn': ('rho0','E1'),
              'dnup': ('rho0','E2'), 'dndn': ('rho0','remainder') }, basis="pp")


specs16x10 = _spamc.build_spam_specs(
    prepStrs=prepStrs,
    effectStrs=effectStrs,
    prep_labels=gs_target.get_prep_labels(),
    effect_labels=gs_target.get_effect_labels() )

specs16 = _spamc.build_spam_specs(
    fiducials16,
    prep_labels=gs_target.get_prep_labels(),
    effect_labels=gs_target.get_effect_labels() )

specs36 = _spamc.build_spam_specs(
    fiducials36,
    prep_labels=gs_target.get_prep_labels(),
    effect_labels=gs_target.get_effect_labels() )

specs = specs16x10 #use smallest specs set as "default"
