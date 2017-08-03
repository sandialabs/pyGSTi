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
from collections import OrderedDict as _OrderedDict

description = "I*I, I*X(pi/2), I*Y(pi/2), X(pi/2)*I, and Y(pi/2)*I gates"

gates = ['Gii', 'Gix','Giy','Gxi','Gyi']

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
    [('Gii',),
     ('Gxi',),
     ('Gyi',),
     ('Gix',),
     ('Giy',),
     ('Gxi', 'Gyi'),
     ('Gix', 'Giy'),
     ('Giy', 'Gxi'),
     ('Gix', 'Gxi'),
     ('Gii', 'Gix'),
     ('Gxi', 'Gyi', 'Gii'),
     ('Gxi', 'Gii', 'Gyi'),
     ('Gxi', 'Gii', 'Gii'),
     ('Gyi', 'Gii', 'Gii'),
     ('Gix', 'Giy', 'Gii'),
     ('Gix', 'Gii', 'Giy'),
     ('Gix', 'Gii', 'Gii'),
     ('Giy', 'Gii', 'Gii'),
     ('Gii', 'Gyi', 'Giy'),
     ('Gix', 'Gyi', 'Giy'),
     ('Gxi', 'Gxi', 'Gii', 'Gyi'),
     ('Gxi', 'Gyi', 'Gyi', 'Gii'),
     ('Gix', 'Gix', 'Gii', 'Giy'),
     ('Gix', 'Giy', 'Giy', 'Gii'),
     ('Gix', 'Gyi', 'Gix', 'Gyi'),
     ('Gyi', 'Gyi', 'Giy', 'Gyi'),
     ('Gix', 'Gix', 'Gxi', 'Gix'),
     ('Giy', 'Gix', 'Gix', 'Gix'),
     ('Gix', 'Gyi', 'Gyi', 'Gyi'),
     ('Gix', 'Gii', 'Gii', 'Gxi'),
     ('Gix', 'Gii', 'Gyi', 'Gii'),
     ('Giy', 'Giy', 'Gxi', 'Gyi', 'Gxi'),
     ('Giy', 'Giy', 'Giy', 'Gix', 'Gxi'),
     ('Gix', 'Gxi', 'Gyi', 'Gxi', 'Giy'),
     ('Gxi', 'Gxi', 'Gyi', 'Gxi', 'Gyi', 'Gyi'),
     ('Gix', 'Gix', 'Giy', 'Gix', 'Giy', 'Giy'),
     ('Gix', 'Giy', 'Gix', 'Giy', 'Gix', 'Gyi'),
     ('Gyi', 'Gii', 'Giy', 'Gxi', 'Gxi', 'Giy'),
     ('Gxi', 'Gix', 'Giy', 'Gxi', 'Giy', 'Gyi'),
     ('Giy', 'Gii', 'Gii', 'Gxi', 'Giy', 'Gxi'),
     ('Gxi', 'Gix', 'Giy', 'Gix', 'Giy', 'Gix'),
     ('Gix', 'Giy', 'Gix', 'Gyi', 'Gxi', 'Gii', 'Gxi'),
     ('Gxi', 'Gyi', 'Gyi', 'Gix', 'Giy', 'Gxi', 'Giy'),
     ('Giy', 'Gii', 'Gyi', 'Gyi', 'Gix', 'Gxi', 'Giy'),
     ('Giy', 'Gyi', 'Gxi', 'Gyi', 'Gix', 'Gix', 'Giy'),
     ('Gxi', 'Giy', 'Gxi', 'Gyi', 'Gix', 'Gii', 'Gxi'),
     ('Gxi', 'Giy', 'Gix', 'Gyi', 'Gix', 'Gix', 'Gii'),
     ('Giy', 'Gii', 'Gii', 'Giy', 'Giy', 'Gii', 'Giy'),
     ('Gxi', 'Gii', 'Giy', 'Gxi', 'Gyi', 'Giy', 'Gii'),
     ('Giy', 'Gxi', 'Gyi', 'Giy', 'Gyi', 'Gxi', 'Gii'),
     ('Gii', 'Gxi', 'Giy', 'Gyi', 'Gyi', 'Gix', 'Gyi'),
     ('Gix', 'Giy', 'Giy', 'Gyi', 'Gii', 'Gxi', 'Giy'),
     ('Gyi', 'Gix', 'Gxi', 'Gyi', 'Gxi', 'Gii', 'Giy'),
     ('Gix', 'Gyi', 'Gii', 'Gix', 'Gix', 'Gxi', 'Gyi'),
     ('Giy', 'Gxi', 'Gix', 'Giy', 'Gyi', 'Giy', 'Gxi'),
     ('Giy', 'Gii', 'Gxi', 'Gxi', 'Gix', 'Gii', 'Gyi', 'Giy'),
     ('Gxi', 'Giy', 'Gix', 'Gyi', 'Gix', 'Gii', 'Gxi', 'Giy'),
     ('Giy', 'Gix', 'Gii', 'Gyi', 'Gii', 'Gyi', 'Gxi', 'Giy'),
     ('Giy', 'Gyi', 'Gix', 'Gix', 'Gxi', 'Gxi', 'Gxi', 'Gyi'),
     ('Gii', 'Gii', 'Gyi', 'Giy', 'Gix', 'Giy', 'Gix', 'Gxi'),
     ('Gxi', 'Gii', 'Gii', 'Gix', 'Giy', 'Gxi', 'Gyi', 'Gix'),
     ('Gyi', 'Gxi', 'Giy', 'Gxi', 'Gix', 'Gxi', 'Gyi', 'Giy'),
     ('Gyi', 'Gii', 'Gix', 'Gyi', 'Gyi', 'Gxi', 'Gix', 'Giy') ] )
    
#Construct the target gateset
gs_target = _setc.build_gateset(
    [4], [('Q0','Q1')],['Gii','Gix','Giy','Gxi','Gyi'],
    [ "I(Q0):I(Q1)", "I(Q0):X(pi/2,Q1)", "I(Q0):Y(pi/2,Q1)", "X(pi/2,Q0):I(Q1)", "Y(pi/2,Q0):I(Q1)" ],
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

clifford_compilation = _OrderedDict()
clifford_compilation['Gc0c0'] = ['Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c1'] = ['Giy', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c2'] = ['Gix', 'Gix', 'Gix', 'Giy', 'Giy', 'Giy', 'Gii']   
clifford_compilation['Gc0c3'] = ['Gix', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c4'] = ['Giy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gix', 'Gii']   
clifford_compilation['Gc0c5'] = ['Gix', 'Giy', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c6'] = ['Giy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c7'] = ['Giy', 'Giy', 'Giy', 'Gix', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c8'] = ['Gix', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c9'] = ['Gix', 'Gix', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c10'] = ['Giy', 'Gix', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c11'] = ['Gix', 'Gix', 'Gix', 'Giy', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c12'] = ['Giy', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c13'] = ['Gix', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c14'] = ['Gix', 'Giy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gix']   
clifford_compilation['Gc0c15'] = ['Giy', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c16'] = ['Gix', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c17'] = ['Gix', 'Giy', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c18'] = ['Giy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gii', 'Gii']   
clifford_compilation['Gc0c19'] = ['Gix', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c20'] = ['Gix', 'Giy', 'Giy', 'Giy', 'Gix', 'Gii', 'Gii']   
clifford_compilation['Gc0c21'] = ['Giy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c22'] = ['Gix', 'Gix', 'Gix', 'Giy', 'Giy', 'Gii', 'Gii']   
clifford_compilation['Gc0c23'] = ['Gix', 'Giy', 'Gix', 'Gix', 'Gix', 'Gii', 'Gii']   
clifford_compilation['Gc1c0'] = ['Gyi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc2c0'] = ['Gxi', 'Gxi', 'Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gii']   
clifford_compilation['Gc3c0'] = ['Gxi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']     
clifford_compilation['Gc4c0'] = ['Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gxi', 'Gii']   
clifford_compilation['Gc5c0'] = ['Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc6c0'] = ['Gyi', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc7c0'] = ['Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gii', 'Gii', 'Gii']    
clifford_compilation['Gc8c0'] = ['Gxi', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc9c0'] = ['Gxi', 'Gxi', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc10c0'] = ['Gyi', 'Gxi', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc11c0'] = ['Gxi', 'Gxi', 'Gxi', 'Gyi', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc12c0'] = ['Gyi', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']    
clifford_compilation['Gc13c0'] = ['Gxi', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc14c0'] = ['Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gxi']   
clifford_compilation['Gc15c0'] = ['Gyi', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii']    
clifford_compilation['Gc16c0'] = ['Gxi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc17c0'] = ['Gxi', 'Gyi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc18c0'] = ['Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gii', 'Gii']   
clifford_compilation['Gc19c0'] = ['Gxi', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc20c0'] = ['Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gii', 'Gii']   
clifford_compilation['Gc21c0'] = ['Gyi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc22c0'] = ['Gxi', 'Gxi', 'Gxi', 'Gyi', 'Gyi', 'Gii', 'Gii']   
clifford_compilation['Gc23c0'] = ['Gxi', 'Gyi', 'Gxi', 'Gxi', 'Gxi', 'Gii', 'Gii']   