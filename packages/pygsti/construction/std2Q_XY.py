from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Variables for working with the 2-qubit gate set containing the gates
I*X(pi/2), I*Y(pi/2), X(pi/2)*I, Y(pi/2)*I, and CPHASE.
"""

from . import gatestringconstruction as _strc
from . import gatesetconstruction as _setc
from . import spamspecconstruction as _spamc

description = "I*X(pi/2), I*Y(pi/2), X(pi/2)*I, Y(pi/2)*I, and CPHASE gates"

gates = ['Gix','Giy','Gxi','Gyi','Gcphase']

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

germs = _strc.gatestring_list(
    [('Gyi',),
     ('Gxi',),
     ('Giy',),
     ('Gix',),
     ('Gxi', 'Gyi'),
     ('Gix', 'Giy'),
     ('Gix', 'Gxi'),
     ('Giy', 'Gyi'),
     ('Gix', 'Gxi', 'Giy'),
     ('Gix', 'Gyi', 'Giy'),
     ('Gyi', 'Gyi', 'Gyi', 'Gxi'),
     ('Giy', 'Giy', 'Giy', 'Gix'),
     ('Gyi', 'Gix', 'Gxi', 'Giy'),
     ('Giy', 'Giy', 'Gxi', 'Gxi'),
     ('Giy', 'Gyi', 'Gyi', 'Gyi'),
     ('Giy', 'Gyi', 'Giy', 'Gyi', 'Gxi'),
     ('Gix', 'Gyi', 'Gix', 'Gyi', 'Giy'),
     ('Giy', 'Gxi', 'Giy', 'Gxi', 'Gyi'),
     ('Gyi', 'Gxi', 'Gyi', 'Gxi', 'Gxi', 'Gxi'),
     ('Gyi', 'Gxi', 'Gyi', 'Gyi', 'Gxi', 'Gxi'),
     ('Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gyi', 'Gxi'),
     ('Gxi', 'Gxi', 'Gyi', 'Gxi', 'Gyi', 'Gyi'),
     ('Giy', 'Gix', 'Giy', 'Gix', 'Gix', 'Gix'),
     ('Giy', 'Gix', 'Giy', 'Giy', 'Gix', 'Gix'),
     ('Giy', 'Giy', 'Giy', 'Gix', 'Giy', 'Gix'),
     ('Gix', 'Gix', 'Giy', 'Gix', 'Giy', 'Giy'),
     ('Gix', 'Gxi', 'Gix', 'Giy', 'Gix', 'Giy'),
     ('Gxi', 'Gxi', 'Gix', 'Gyi', 'Giy', 'Gix'),
     ('Gyi', 'Gyi', 'Gix', 'Gxi', 'Gyi', 'Gxi'),
     ('Giy', 'Gxi', 'Gix', 'Giy', 'Gyi', 'Gxi'),
     ('Gyi', 'Giy', 'Gix', 'Gyi', 'Gix', 'Gxi'),
     ('Gix', 'Gxi', 'Giy', 'Gix', 'Gyi', 'Gix'),
     ('Gyi', 'Giy', 'Gyi', 'Gix', 'Gix', 'Gxi'),
     ('Giy', 'Gxi', 'Gyi', 'Gxi', 'Gix', 'Gxi'),
     ('Giy', 'Gyi', 'Giy', 'Giy', 'Gxi', 'Gix'),
     ('Gyi', 'Giy', 'Giy', 'Giy', 'Gxi', 'Gxi'),
     ('Gix', 'Gxi', 'Gix', 'Giy', 'Gyi', 'Giy', 'Gxi'),
     ('Gyi', 'Gix', 'Gyi', 'Giy', 'Giy', 'Gyi', 'Gxi'),
     ('Gxi', 'Gyi', 'Gyi', 'Gix', 'Giy', 'Giy', 'Gix'),
     ('Gxi', 'Gxi', 'Gyi', 'Gyi', 'Gix', 'Gix', 'Gxi'),
     ('Gyi', 'Gyi', 'Giy', 'Gxi', 'Giy', 'Giy', 'Giy'),
     ('Giy', 'Gyi', 'Gyi', 'Giy', 'Giy', 'Giy', 'Gyi', 'Gxi'),
     ('Gxi', 'Gyi', 'Gix', 'Giy', 'Gyi', 'Giy', 'Giy', 'Gix'),
     ('Gix', 'Gxi', 'Gix', 'Gyi', 'Gxi', 'Gyi', 'Gyi', 'Gix'),
     ('Gix', 'Gxi', 'Gyi', 'Gix', 'Giy', 'Gyi', 'Gyi', 'Gxi'),
     ('Gxi', 'Gyi', 'Gyi', 'Giy', 'Gxi', 'Gxi', 'Gxi', 'Giy'),
     ('Gyi', 'Gxi', 'Gix', 'Gxi', 'Gxi', 'Gyi', 'Gyi', 'Giy'),
     ('Gix', 'Giy', 'Gxi', 'Giy', 'Gxi', 'Gxi', 'Gyi', 'Giy'),
     ('Giy', 'Gix', 'Gyi', 'Gyi', 'Gix', 'Gxi', 'Giy', 'Giy') ])
    
#Construct the target gateset
gs_target = _setc.build_gateset(
    [4], [('Q0','Q1')],['Gix','Giy','Gxi','Gyi'],
    [ "I(Q0):X(pi/2,Q1)", "I(Q0):Y(pi/2,Q1)", "X(pi/2,Q0):I(Q1)", "Y(pi/2,Q0):I(Q1)" ],
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
